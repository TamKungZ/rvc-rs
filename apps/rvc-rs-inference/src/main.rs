#![forbid(unsafe_code)]

use eframe::egui;
use rfd::FileDialog;
use rvc_rs_core::ComputeDevice;
use rvc_rs_engine::{Engine, EngineConfig, EngineState, ModelFiles, OfflineJob, OfflineReport};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1_100.0, 760.0])
            .with_min_inner_size([860.0, 620.0]),
        ..Default::default()
    };
    eframe::run_native(
        "RVC.rs Inference",
        options,
        Box::new(|context| Ok(Box::new(InferenceApp::new(context)))),
    )
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
enum ConversionMode {
    #[default]
    Offline,
    Realtime,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct SavedForm {
    mode: ConversionMode,
    config: EngineConfig,
    checkpoint: String,
    contentvec: String,
    rmvpe: String,
    index: String,
    input_audio: String,
    output_audio: String,
}

struct InferenceApp {
    engine: Engine,
    form: SavedForm,
    messages: Vec<String>,
    worker: Option<Receiver<Result<OfflineReport, String>>>,
}

impl InferenceApp {
    fn new(context: &eframe::CreationContext<'_>) -> Self {
        context.egui_ctx.set_visuals(egui::Visuals::dark());
        let form = context
            .storage
            .and_then(|storage| eframe::get_value(storage, eframe::APP_KEY))
            .unwrap_or_default();

        let mut app = Self {
            engine: Engine::new(),
            form,
            messages: vec![
                "Native direction: direct RVC .pth and optional FAISS .index.".to_owned(),
                "RVC v2 offline conversion is available; real-time audio is the next milestone."
                    .to_owned(),
            ],
            worker: None,
        };
        app.sync_engine();
        app
    }

    fn sync_engine(&mut self) {
        if let Err(error) = self.engine.set_config(self.form.config.clone()) {
            self.push_message(error.to_string());
        }
        if !self.form.checkpoint.is_empty() {
            self.engine.set_model(self.model_files());
        }
    }

    fn model_files(&self) -> ModelFiles {
        ModelFiles {
            checkpoint: PathBuf::from(&self.form.checkpoint),
            contentvec: (!self.form.contentvec.is_empty())
                .then(|| PathBuf::from(&self.form.contentvec)),
            rmvpe: (!self.form.rmvpe.is_empty()).then(|| PathBuf::from(&self.form.rmvpe)),
            index: (!self.form.index.is_empty()).then(|| PathBuf::from(&self.form.index)),
        }
    }

    fn offline_job(&self) -> OfflineJob {
        OfflineJob {
            input_audio: PathBuf::from(&self.form.input_audio),
            output_audio: PathBuf::from(&self.form.output_audio),
        }
    }

    fn offline_validation(&self) -> Result<(), String> {
        self.form
            .config
            .validate()
            .map_err(|error| error.to_string())?;
        self.model_files()
            .validate()
            .map_err(|error| error.to_string())?;
        self.offline_job()
            .validate()
            .map_err(|error| error.to_string())
    }

    fn push_message(&mut self, message: String) {
        if self.messages.last() != Some(&message) {
            self.messages.push(message);
        }
        const MAX_MESSAGES: usize = 100;
        if self.messages.len() > MAX_MESSAGES {
            self.messages.drain(..self.messages.len() - MAX_MESSAGES);
        }
    }

    fn poll_worker(&mut self) {
        let result = self
            .worker
            .as_ref()
            .and_then(|worker| match worker.try_recv() {
                Ok(result) => Some(result),
                Err(TryRecvError::Disconnected) => {
                    Some(Err("conversion worker stopped without a result".to_owned()))
                }
                Err(TryRecvError::Empty) => None,
            });
        if let Some(result) = result {
            self.engine.finish_offline(&result);
            match &result {
                Ok(report) => self.push_message(format!(
                    "Completed: {} samples at {} Hz in {:.2}s (model {:.2}s) -> {}",
                    report.samples,
                    report.sample_rate,
                    report.elapsed.as_secs_f32(),
                    report.inference_time.as_secs_f32(),
                    report.output_audio.display()
                )),
                Err(error) => self.push_message(error.clone()),
            }
            self.worker = None;
        }
    }

    fn top_bar(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("RVC.rs Inference");
            ui.add_space(8.0);
            ui.label(egui::RichText::new("native Rust voice conversion").weak());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let state = self.engine.state();
                let color = match state {
                    EngineState::Empty => egui::Color32::GRAY,
                    EngineState::Configured => egui::Color32::YELLOW,
                    EngineState::Preparing => egui::Color32::LIGHT_BLUE,
                    EngineState::Ready => egui::Color32::LIGHT_GREEN,
                    EngineState::Running => egui::Color32::LIGHT_BLUE,
                    EngineState::Failed(_) => egui::Color32::LIGHT_RED,
                };
                ui.label(egui::RichText::new(state.label()).color(color).strong());
            });
        });
    }

    fn mode_selector(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.form.mode, ConversionMode::Offline, "Offline file");
            ui.selectable_value(
                &mut self.form.mode,
                ConversionMode::Realtime,
                "Real-time microphone",
            );
        });
    }

    fn model_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Voice model");
        path_picker(ui, "RVC model", &mut self.form.checkpoint, &["pth"], false);
        path_picker(ui, "ContentVec", &mut self.form.contentvec, &["pt"], false);
        path_picker(
            ui,
            "Retrieval index",
            &mut self.form.index,
            &["index"],
            false,
        );
        ui.label(
            egui::RichText::new(concat!(
                "The production path loads .pth tensors directly into Candle and the ",
                "optional IVF-Flat .index into a preallocated native search workspace."
            ))
            .small()
            .weak(),
        );
    }

    fn settings_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inference settings");
        egui::Grid::new("inference_settings")
            .num_columns(2)
            .spacing([18.0, 10.0])
            .show(ui, |ui| {
                ui.label("Compute device");
                egui::ComboBox::from_id_salt("compute_device")
                    .selected_text(device_label(self.form.config.device))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.form.config.device,
                            ComputeDevice::Auto,
                            "Auto",
                        );
                        ui.selectable_value(
                            &mut self.form.config.device,
                            ComputeDevice::Cpu,
                            "CPU",
                        );
                        ui.selectable_value(
                            &mut self.form.config.device,
                            ComputeDevice::Cuda(0),
                            "CUDA 0",
                        );
                        ui.selectable_value(
                            &mut self.form.config.device,
                            ComputeDevice::Metal(0),
                            "Metal 0",
                        );
                    });
                ui.end_row();

                ui.label("Pitch shift");
                ui.add(
                    egui::Slider::new(&mut self.form.config.pitch_shift, -24..=24).suffix(" st"),
                );
                ui.end_row();

                ui.label("Retrieval rate");
                ui.add(
                    egui::Slider::new(&mut self.form.config.retrieval_rate, 0.0..=1.0)
                        .fixed_decimals(2),
                );
                ui.end_row();

                ui.label("Speaker ID");
                ui.add(egui::DragValue::new(&mut self.form.config.speaker_id).range(0..=255));
                ui.end_row();

                ui.label("Chunk");
                ui.add(egui::Slider::new(&mut self.form.config.chunk_ms, 20..=1_000).suffix(" ms"));
                ui.end_row();

                ui.label("Crossfade");
                ui.add(
                    egui::Slider::new(&mut self.form.config.crossfade_ms, 0..=250).suffix(" ms"),
                );
                ui.end_row();
            });
    }

    fn offline_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Offline conversion");
        path_picker(
            ui,
            "Input audio",
            &mut self.form.input_audio,
            &["wav"],
            false,
        );
        path_picker(
            ui,
            "Output WAV",
            &mut self.form.output_audio,
            &["wav"],
            true,
        );

        let validation = self.offline_validation();
        if let Err(message) = &validation {
            ui.label(egui::RichText::new(message).color(egui::Color32::LIGHT_RED));
        } else {
            ui.label(
                egui::RichText::new("Paths and settings are valid.")
                    .color(egui::Color32::LIGHT_GREEN),
            );
        }

        ui.horizontal(|ui| {
            if ui.button("Backend doctor").clicked() {
                self.sync_engine();
                match self.engine.doctor() {
                    Ok(values) => {
                        self.push_message(format!("Backend tensor round trip passed: {values:?}"))
                    }
                    Err(error) => self.push_message(error.to_string()),
                }
            }

            if ui.button("Prepare native model").clicked() {
                self.sync_engine();
                match self.engine.prepare_native() {
                    Ok(report) => self.push_message(format!(
                        "Prepared {} tensors, {}-D, {} Hz, {} speaker(s), index={:?}",
                        report.tensor_count,
                        report.feature_dimension,
                        report.sample_rate,
                        report.speaker_count,
                        report.index_vectors
                    )),
                    Err(error) => self.push_message(error.to_string()),
                }
            }

            let running = self.worker.is_some();
            let run = ui.add_enabled(
                validation.is_ok() && !running,
                egui::Button::new(if running {
                    "Converting…"
                } else {
                    "Start offline conversion"
                }),
            );
            if run.clicked() {
                self.sync_engine();
                let job = self.offline_job();
                match self.engine.begin_offline(&job) {
                    Ok(task) => {
                        let (sender, receiver) = mpsc::channel();
                        std::thread::spawn(move || {
                            let result = task.run().map_err(|error| error.to_string());
                            let _ = sender.send(result);
                        });
                        self.worker = Some(receiver);
                        self.push_message(
                            "Offline conversion started in the background.".to_owned(),
                        );
                    }
                    Err(error) => self.push_message(error.to_string()),
                }
            }
            if running {
                ui.spinner();
            }
        });
    }

    fn realtime_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Real-time microphone");
        ui.label("The native model path is active; CPAL capture/playback integration is not enabled yet.");
        ui.add_enabled(false, egui::Button::new("Start real-time conversion"));
        ui.add_space(8.0);
        ui.label(
            egui::RichText::new(concat!(
                "Implemented: direct checkpoint loading, ContentVec, DSP F0, in-memory retrieval, ",
                "native RVC synthesis, and allocation-free SOLA. Next: CPAL worker."
            ))
            .weak(),
        );
    }

    fn log_section(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.heading("Activity");
            if ui.small_button("Clear").clicked() {
                self.messages.clear();
            }
        });
        egui::ScrollArea::vertical()
            .max_height(150.0)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for message in &self.messages {
                    ui.monospace(message);
                }
            });
    }
}

impl eframe::App for InferenceApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_worker();
        if self.worker.is_some() {
            ui.ctx().request_repaint_after(Duration::from_millis(50));
        }
        egui::CentralPanel::default().show(ui, |ui| {
            self.top_bar(ui);
            ui.separator();
            self.mode_selector(ui);
            ui.add_space(8.0);

            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.columns(2, |columns| {
                    self.model_section(&mut columns[0]);
                    self.settings_section(&mut columns[1]);
                });
                ui.separator();
                match self.form.mode {
                    ConversionMode::Offline => self.offline_section(ui),
                    ConversionMode::Realtime => self.realtime_section(ui),
                }
                ui.separator();
                self.log_section(ui);
            });
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, &self.form);
    }
}

fn path_picker(ui: &mut egui::Ui, label: &str, path: &mut String, extensions: &[&str], save: bool) {
    ui.horizontal(|ui| {
        ui.label(label);
        ui.add(egui::TextEdit::singleline(path).desired_width(260.0));
        if ui.button("Browse…").clicked() {
            let mut dialog = FileDialog::new().add_filter(label, extensions);
            let selected = if save {
                dialog = dialog.set_file_name("converted.wav");
                dialog.save_file()
            } else {
                dialog.pick_file()
            };
            if let Some(selected) = selected {
                *path = selected.display().to_string();
            }
        }
    });
}

fn device_label(device: ComputeDevice) -> String {
    match device {
        ComputeDevice::Auto => "Auto".to_owned(),
        ComputeDevice::Cpu => "CPU".to_owned(),
        ComputeDevice::Cuda(index) => format!("CUDA {index}"),
        ComputeDevice::Metal(index) => format!("Metal {index}"),
    }
}
