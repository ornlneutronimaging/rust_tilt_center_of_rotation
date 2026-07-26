//! Tilt & center-of-rotation estimation for CT projection stacks.
//!
//! The GUI binary (`main.rs`) is a thin shell around these modules; the
//! math and the HDF5 I/O live here so they can be tested without a display.

pub mod algorithms;
pub mod app;
pub mod logger;
pub mod npy;
pub mod pairs;
pub mod recon;
pub mod stack;
