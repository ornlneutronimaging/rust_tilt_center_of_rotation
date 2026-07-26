//! Standalone tilt & center-of-rotation tool.
//!
//! Usage: `tilt_center_of_rotation [checkpoint.h5] [--called-from-app]`
//! With a path the stack loads immediately; without one the window opens
//! empty and the stack is loaded from the 📂 button.

use tilt_center_of_rotation::app::TiltCorApp;

fn main() -> eframe::Result<()> {
    let path = std::env::args()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .map(std::path::PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 860.0])
            .with_title("Tilt & Center of Rotation"),
        ..Default::default()
    };
    eframe::run_native(
        "tilt_center_of_rotation",
        options,
        Box::new(move |cc| {
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            Ok(Box::new(TiltCorApp::new(path)))
        }),
    )
}
