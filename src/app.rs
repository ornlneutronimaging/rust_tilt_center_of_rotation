//! The GUI: load a checkpoint, pick the opposite pair(s), run the
//! estimators, inspect the fit, then apply the tilt and save the corrected
//! stack (plus the center of rotation) back into the HDF5.

use crate::algorithms::{self, Estimate, Params};
use crate::pairs::{self, Pair};
use crate::stack::{self, Stack};
use egui::{Color32, RichText};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::Duration;

/// What the central image shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Proj0,
    Proj180Flipped,
    /// |0° − 180° mirrored about the adopted axis| — flat where the axis
    /// is right.
    Difference,
}

impl View {
    fn label(self) -> &'static str {
        match self {
            View::Proj0 => "0° projection",
            View::Proj180Flipped => "180° flipped",
            View::Difference => "difference about the axis",
        }
    }
}

pub struct TiltCorApp {
    stack: Option<Arc<Stack>>,
    load_job: Option<Receiver<Result<Stack, String>>>,
    load_error: Option<String>,

    tol_deg: f64,
    pairs: Vec<Pair>,
    pair_idx: usize,

    // Estimation parameters (mirrors algorithms::Params).
    y_top: usize,
    y_bottom: usize,
    ystep: usize,
    band: usize,
    max_shift: usize,

    est_job: Option<Receiver<Result<(Estimate, Option<f64>), String>>>,
    estimates: Vec<Estimate>,
    global_cor: Option<f64>,
    est_error: Option<String>,
    /// Index into `estimates` of the result the correction will use.
    adopted: Option<usize>,

    /// 1 = show the spinner this frame, 2 = rotate + start the save next
    /// frame (so the spinner is on screen during the heavy work).
    apply_pending: u8,
    save_job: Option<Receiver<Result<String, String>>>,
    save_status: Option<Result<String, String>>,

    view: View,
    /// Cached texture, keyed by (pair index, view, adopted estimate index).
    tex: Option<((usize, u8, usize), egui::TextureHandle)>,
}

impl TiltCorApp {
    pub fn new(path: Option<PathBuf>) -> Self {
        let mut app = Self {
            stack: None,
            load_job: None,
            load_error: None,
            tol_deg: 0.5,
            pairs: Vec::new(),
            pair_idx: 0,
            y_top: 0,
            y_bottom: 0,
            ystep: 1,
            band: 5,
            max_shift: 100,
            est_job: None,
            estimates: Vec::new(),
            global_cor: None,
            est_error: None,
            adopted: None,
            apply_pending: 0,
            save_job: None,
            save_status: None,
            view: View::Difference,
            tex: None,
        };
        if let Some(path) = path {
            app.start_load(path);
        }
        app
    }

    fn start_load(&mut self, path: PathBuf) {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(stack::load(&path));
        });
        self.load_job = Some(rx);
        self.load_error = None;
    }

    /// Fold a freshly loaded stack in: reset the parameters and the pair
    /// list to its geometry.
    fn adopt_stack(&mut self, stack: Stack) {
        let params = Params::defaults_for(stack.height, stack.width);
        self.y_top = params.y_top;
        self.y_bottom = params.y_bottom;
        self.ystep = params.ystep;
        self.band = params.band;
        self.max_shift = params.max_shift;
        self.pairs = pairs::opposite_pairs(&stack.angles_deg, self.tol_deg);
        self.pair_idx = 0;
        self.estimates.clear();
        self.global_cor = None;
        self.adopted = None;
        self.est_error = None;
        self.save_status = None;
        self.tex = None;
        self.stack = Some(Arc::new(stack));
    }

    fn params(&self) -> Params {
        Params {
            y_top: self.y_top,
            y_bottom: self.y_bottom,
            ystep: self.ystep,
            band: self.band,
            max_shift: self.max_shift,
        }
    }

    fn start_estimate(&mut self, all_pairs: bool) {
        let Some(stack) = self.stack.clone() else {
            return;
        };
        let Some(&pair) = self.pairs.get(self.pair_idx) else {
            return;
        };
        let pairs = self.pairs.clone();
        let params = self.params();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = if all_pairs {
                algorithms::multi_pair(&stack, &pairs, &params)
            } else {
                algorithms::single_pair(&stack, &pair, &params)
            }
            .map(|est| {
                let global = algorithms::global_cor(&stack, &pair, &params).ok();
                (est, global)
            });
            let _ = tx.send(result);
        });
        self.est_job = Some(rx);
        self.est_error = None;
    }

    /// The rotation + save, run once the spinner frame has been drawn.
    fn apply_and_save(&mut self) {
        let Some(k) = self.adopted else { return };
        let Some(est) = self.estimates.get(k).cloned() else {
            return;
        };
        let Some(stack_arc) = &mut self.stack else {
            return;
        };
        let Some(stack) = Arc::get_mut(stack_arc) else {
            self.save_status = Some(Err(
                "an estimation is still running — wait for it to finish".to_owned()
            ));
            return;
        };
        algorithms::apply_tilt(stack, est.tilt_deg);
        stack.center_of_rotation = Some(est.cor);
        let record = serde_json::json!({
            "corrected_tilt_deg": est.tilt_deg,
            "center_of_rotation": est.cor,
            "method": est.method,
            "r2": est.r2,
            "date": chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            "user": std::env::var("USER").unwrap_or_default(),
        })
        .to_string();
        let stack = stack_arc.clone();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = stack::save(&stack, est.cor, &record).map(|()| {
                format!(
                    "tilt of {:+.3}° corrected, center of rotation {:.2} px — saved into {}",
                    est.tilt_deg,
                    est.cor,
                    stack.path.display()
                )
            });
            let _ = tx.send(result);
        });
        self.save_job = Some(rx);
        // The stack changed: previous estimates no longer describe it.
        self.estimates.clear();
        self.global_cor = None;
        self.adopted = None;
        self.tex = None;
    }

    fn poll_jobs(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.load_job {
            match rx.try_recv() {
                Ok(Ok(stack)) => {
                    self.load_job = None;
                    self.adopt_stack(stack);
                }
                Ok(Err(e)) => {
                    self.load_job = None;
                    self.load_error = Some(e);
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(200)),
            }
        }
        if let Some(rx) = &self.est_job {
            match rx.try_recv() {
                Ok(Ok((est, global))) => {
                    self.est_job = None;
                    self.global_cor = global;
                    self.estimates.push(est);
                    self.adopted = Some(self.estimates.len() - 1);
                    self.tex = None;
                }
                Ok(Err(e)) => {
                    self.est_job = None;
                    self.est_error = Some(e);
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(200)),
            }
        }
        if let Some(rx) = &self.save_job {
            match rx.try_recv() {
                Ok(result) => {
                    self.save_job = None;
                    self.save_status = Some(result);
                }
                Err(_) => ctx.request_repaint_after(Duration::from_millis(300)),
            }
        }
    }

    /// The plane the current view shows, as 8-bit grayscale with a 1–99
    /// percentile window.
    fn view_image(&self, stack: &Stack) -> Option<egui::ColorImage> {
        let pair = self.pairs.get(self.pair_idx)?;
        let (w, h) = (stack.width, stack.height);
        let axis = self.adopted.and_then(|k| self.estimates.get(k));
        let pixels: Vec<f32> = match self.view {
            View::Proj0 => stack.plane(pair.i).to_vec(),
            View::Proj180Flipped => {
                let p = stack.plane(pair.j);
                let mut out = Vec::with_capacity(h * w);
                for y in 0..h {
                    let row = &p[y * w..(y + 1) * w];
                    out.extend(row.iter().rev());
                }
                out
            }
            View::Difference => {
                let p0 = stack.plane(pair.i);
                let p180 = stack.plane(pair.j);
                let mut out = Vec::with_capacity(h * w);
                for y in 0..h {
                    let c = match axis {
                        Some(e) => e.slope * y as f64 + e.intercept,
                        None => (w as f64 - 1.0) / 2.0,
                    };
                    for x in 0..w {
                        let xm = (2.0 * c - x as f64).round();
                        let m = if xm >= 0.0 && xm < w as f64 {
                            p180[y * w + xm as usize]
                        } else {
                            p0[y * w + x]
                        };
                        out.push((p0[y * w + x] - m).abs());
                    }
                }
                out
            }
        };
        // Percentile window on a subsample.
        let mut sample: Vec<f32> = pixels
            .iter()
            .step_by((pixels.len() / 40_000).max(1))
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        sample.sort_by(|a, b| a.total_cmp(b));
        let (lo, hi) = if sample.is_empty() {
            (0.0, 1.0)
        } else {
            let lo = sample[sample.len() / 100];
            let hi = sample[sample.len() - 1 - sample.len() / 100];
            if hi > lo { (lo, hi) } else { (lo, lo + 1.0) }
        };
        let gray: Vec<u8> = pixels
            .iter()
            .map(|&v| (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0) as u8)
            .collect();
        Some(egui::ColorImage::from_gray([w, h], &gray))
    }

    fn left_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        // File row.
        ui.horizontal(|ui| {
            if ui.button("📂 Load HDF5…").clicked() {
                let mut dialog = rfd::FileDialog::new()
                    .set_title("Select a projections HDF5 (ct_reconstruction format)")
                    .add_filter("HDF5", &["h5", "hdf5"]);
                if let Some(stack) = &self.stack
                    && let Some(dir) = stack.path.parent()
                {
                    dialog = dialog.set_directory(dir);
                }
                if let Some(path) = dialog.pick_file() {
                    self.start_load(path);
                }
            }
            if self.load_job.is_some() {
                ui.spinner();
                ui.label("loading…");
            }
        });
        if let Some(e) = &self.load_error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        let Some(stack) = self.stack.clone() else {
            ui.add_space(8.0);
            ui.label(RichText::new("no stack loaded yet").weak());
            return;
        };
        ui.label(
            RichText::new(format!(
                "{} — {} projections, {}×{}",
                stack
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                stack.n,
                stack.height,
                stack.width
            ))
            .strong(),
        );
        ui.label(
            RichText::new(format!(
                "center of rotation in the file: {}",
                match stack.center_of_rotation {
                    Some(c) => format!("{c:.2} px"),
                    None => "none".to_owned(),
                }
            ))
            .weak()
            .size(12.0),
        );
        if !stack.previous_corrections.is_empty() {
            ui.colored_label(
                Color32::from_rgb(240, 180, 60),
                format!(
                    "⚠ this file was already corrected {} time(s) by this tool",
                    stack.previous_corrections.len()
                ),
            );
        }
        ui.separator();

        // Opposite pairs.
        ui.label(RichText::new("Opposite (0°/180°) pairs").strong());
        ui.horizontal(|ui| {
            ui.label("tolerance:");
            let drag = egui::DragValue::new(&mut self.tol_deg)
                .speed(0.05)
                .range(0.01..=5.0)
                .suffix("°");
            if ui.add(drag).changed() {
                self.pairs = pairs::opposite_pairs(&stack.angles_deg, self.tol_deg);
                self.pair_idx = 0;
                self.tex = None;
            }
            ui.label(
                RichText::new(format!("{} pair(s)", self.pairs.len()))
                    .weak()
                    .size(12.0),
            );
        });
        if self.pairs.is_empty() {
            ui.colored_label(
                Color32::LIGHT_RED,
                "no opposite pairs — raise the tolerance, or the scan does not \
                 cover 180°",
            );
            return;
        }
        let selected = self.pairs[self.pair_idx.min(self.pairs.len() - 1)];
        egui::ComboBox::from_id_salt("pair_pick")
            .selected_text(selected.label())
            .width(300.0)
            .show_ui(ui, |ui| {
                for (k, p) in self.pairs.iter().enumerate() {
                    if ui
                        .selectable_label(k == self.pair_idx, p.label())
                        .clicked()
                    {
                        self.pair_idx = k;
                        self.tex = None;
                    }
                }
            });
        ui.separator();

        // Parameters.
        ui.label(RichText::new("Row sampling").strong());
        let h = stack.height;
        egui::Grid::new("params_grid").num_columns(2).show(ui, |ui| {
            ui.label("rows from / to:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.y_top).range(0..=h - 2));
                ui.add(egui::DragValue::new(&mut self.y_bottom).range(1..=h - 1));
            });
            ui.end_row();
            ui.label("every / band:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.ystep).range(1..=h / 2));
                ui.add(egui::DragValue::new(&mut self.band).range(1..=50));
            });
            ui.end_row();
            ui.label("max shift (px):");
            ui.add(egui::DragValue::new(&mut self.max_shift).range(2..=stack.width / 2));
            ui.end_row();
        });
        self.y_bottom = self.y_bottom.clamp(self.y_top + 1, h - 1);
        ui.add_space(6.0);

        // Run.
        let busy = self.est_job.is_some();
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!busy, egui::Button::new("▶ Selected pair"))
                .on_hover_text(
                    "per-row registration of the selected 0/180 pair (sub-pixel), \
                     line fit → tilt and center",
                )
                .clicked()
            {
                self.start_estimate(false);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("▶ All pairs (consensus)"))
                .on_hover_text(
                    "the same, with the match error summed over every opposite \
                     pair — slower, robust to a bad pair",
                )
                .clicked()
            {
                self.start_estimate(true);
            }
            if busy {
                ui.spinner();
            }
        });
        if let Some(e) = &self.est_error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        ui.separator();

        // Results.
        ui.label(RichText::new("Results (choose the one to apply)").strong());
        if self.estimates.is_empty() {
            ui.label(RichText::new("no estimate yet").weak());
        }
        let mut adopt: Option<usize> = None;
        for (k, est) in self.estimates.iter().enumerate() {
            let selected = self.adopted == Some(k);
            let text = format!(
                "tilt {:+.3}°, center {:.2} px  (r² {:.3})\n{}",
                est.tilt_deg, est.cor, est.r2, est.method
            );
            if ui.selectable_label(selected, text).clicked() {
                adopt = Some(k);
            }
        }
        if let Some(k) = adopt {
            self.adopted = Some(k);
            self.tex = None;
        }
        if let Some(g) = self.global_cor {
            ui.label(
                RichText::new(format!(
                    "cross-check — whole-image single shift: center {g:.2} px"
                ))
                .weak()
                .size(12.0),
            );
        }
        ui.separator();

        // Apply.
        let can_apply = self.adopted.is_some()
            && self.apply_pending == 0
            && self.save_job.is_none()
            && !busy;
        if ui
            .add_enabled(can_apply, egui::Button::new("💾 Apply the tilt & save into the HDF5"))
            .on_hover_text(
                "rotates every projection by the opposite of the adopted tilt \
                 (bilinear), writes the projections and the center of rotation \
                 back into this file, and records the correction in the metadata",
            )
            .on_disabled_hover_text("run an estimation and select a result first")
            .clicked()
        {
            self.apply_pending = 1;
        }
        if self.apply_pending > 0 || self.save_job.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(if self.apply_pending > 0 {
                    "rotating the projections…"
                } else {
                    "writing the HDF5…"
                });
            });
        }
        match &self.save_status {
            Some(Ok(msg)) => {
                ui.colored_label(Color32::from_rgb(120, 200, 120), msg);
            }
            Some(Err(e)) => {
                ui.colored_label(Color32::LIGHT_RED, e);
            }
            None => {}
        }
    }

    fn central_panel(&mut self, ui: &mut egui::Ui) {
        let Some(stack) = self.stack.clone() else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new(
                        "Load a projections HDF5 (a ct_reconstruction checkpoint) \
                         to estimate the rotation-axis tilt and center",
                    )
                    .weak(),
                );
            });
            return;
        };
        if self.pairs.is_empty() {
            return;
        }
        ui.horizontal(|ui| {
            for view in [View::Proj0, View::Proj180Flipped, View::Difference] {
                if ui
                    .selectable_label(self.view == view, view.label())
                    .clicked()
                {
                    self.view = view;
                    self.tex = None;
                }
            }
            ui.label(
                RichText::new("the red line is the adopted rotation axis")
                    .weak()
                    .size(11.0),
            );
        });

        let key = (
            self.pair_idx,
            self.view as u8,
            self.adopted.map(|k| k + 1).unwrap_or(0),
        );
        if self.tex.as_ref().map(|(k, _)| *k) != Some(key)
            && let Some(image) = self.view_image(&stack)
        {
            let tex = ui
                .ctx()
                .load_texture("view", image, egui::TextureOptions::LINEAR);
            self.tex = Some((key, tex));
        }

        let plot_height = 220.0;
        let image_height = (ui.available_height() - plot_height - 30.0).max(120.0);
        if let Some((_, tex)) = &self.tex {
            let response = ui.add(
                egui::Image::from_texture(tex)
                    .max_height(image_height)
                    .maintain_aspect_ratio(true)
                    .shrink_to_fit(),
            );
            // The adopted axis, drawn over the image.
            if let Some(est) = self.adopted.and_then(|k| self.estimates.get(k)) {
                let rect = response.rect;
                let (w, h) = (stack.width as f32, stack.height as f32);
                let to_screen = |col: f64, row: f64| {
                    egui::pos2(
                        rect.left() + (col as f32 / (w - 1.0)) * rect.width(),
                        rect.top() + (row as f32 / (h - 1.0)) * rect.height(),
                    )
                };
                let flip = matches!(self.view, View::Proj180Flipped);
                let col_at = |row: f64| {
                    let c = est.slope * row + est.intercept;
                    if flip { stack.width as f64 - 1.0 - c } else { c }
                };
                ui.painter().line_segment(
                    [
                        to_screen(col_at(0.0), 0.0),
                        to_screen(col_at(stack.height as f64 - 1.0), stack.height as f64 - 1.0),
                    ],
                    egui::Stroke::new(1.5, Color32::from_rgb(230, 70, 70)),
                );
            }
        }

        // Per-row axis columns + fit.
        if let Some(est) = self.adopted.and_then(|k| self.estimates.get(k)) {
            let used: Vec<[f64; 2]> = est
                .rows
                .iter()
                .filter(|r| r.used)
                .map(|r| [r.row, r.cor])
                .collect();
            let dropped: Vec<[f64; 2]> = est
                .rows
                .iter()
                .filter(|r| !r.used)
                .map(|r| [r.row, r.cor])
                .collect();
            let fit = vec![
                [0.0, est.intercept],
                [
                    stack.height as f64 - 1.0,
                    est.slope * (stack.height as f64 - 1.0) + est.intercept,
                ],
            ];
            egui_plot::Plot::new("axis_fit")
                .height(plot_height)
                .x_axis_label("row (px)")
                .y_axis_label("axis column (px)")
                .legend(egui_plot::Legend::default())
                .show(ui, |plot| {
                    plot.points(
                        egui_plot::Points::new("bands used", used)
                            .radius(2.5)
                            .color(Color32::from_rgb(120, 200, 120)),
                    );
                    if !dropped.is_empty() {
                        plot.points(
                            egui_plot::Points::new("dropped by the fit", dropped)
                                .radius(2.5)
                                .color(Color32::from_rgb(230, 70, 70)),
                        );
                    }
                    plot.line(
                        egui_plot::Line::new("fit", fit)
                            .color(Color32::from_rgb(100, 160, 230)),
                    );
                });
        } else {
            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "run an estimation to see the per-row axis positions and the fit here",
                )
                .weak(),
            );
        }
    }
}

impl eframe::App for TiltCorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_jobs(&ctx);
        // The two-step apply: draw one spinner frame, then do the work.
        match self.apply_pending {
            1 => {
                self.apply_pending = 2;
                ctx.request_repaint();
            }
            2 => {
                self.apply_pending = 0;
                self.apply_and_save();
            }
            _ => {}
        }
        egui::Panel::left("controls")
            .resizable(true)
            .default_size(380.0)
            .min_size(320.0)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.left_panel(ui));
            });
        egui::CentralPanel::default().show(ui, |ui| self.central_panel(ui));
    }
}
