#![forbid(unsafe_code)]

use rvc_rs_core::ComputeDevice;
use rvc_rs_engine::{Engine, EngineConfig, ModelFiles, OfflineJob};
use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    match arguments.next().as_deref() {
        Some("doctor") => {
            let device = parse_device(arguments.next().as_deref().unwrap_or("auto"))?;
            reject_extra(arguments)?;
            let mut engine = Engine::new();
            engine.set_config(EngineConfig {
                device,
                ..EngineConfig::default()
            })?;
            let values = engine.doctor()?;
            println!("backend tensor round trip: {values:?}");
            println!("backend smoke test passed");
            Ok(())
        }
        Some("validate-offline") => {
            let checkpoint = required(&mut arguments, "model.pth")?;
            let index = optional_path(required(&mut arguments, "model.index or -")?);
            let input_audio = required(&mut arguments, "input audio")?;
            let output_audio = required(&mut arguments, "output.wav")?;
            reject_extra(arguments)?;

            let mut engine = Engine::new();
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(checkpoint),
                contentvec: None,
                rmvpe: None,
                index,
            });
            engine.validate_offline(&OfflineJob {
                input_audio: PathBuf::from(input_audio),
                output_audio: PathBuf::from(output_audio),
            })?;
            println!("offline job paths and configuration are valid");
            println!("native .pth/.index paths are valid");
            Ok(())
        }
        Some("prepare-native") => {
            let model = required(&mut arguments, "model.pth")?;
            let index = optional_path(required(&mut arguments, "model.index or -")?);
            let device = parse_device(arguments.next().as_deref().unwrap_or("auto"))?;
            reject_extra(arguments)?;

            let mut engine = Engine::new();
            engine.set_config(EngineConfig {
                device,
                retrieval_rate: if index.is_some() { 0.75 } else { 0.0 },
                ..EngineConfig::default()
            })?;
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(model),
                contentvec: None,
                rmvpe: None,
                index,
            });
            let report = engine.prepare_native()?;
            println!(
                "loaded {} generator tensors: {}-D features, {} Hz, {} speaker(s), f0={}",
                report.tensor_count,
                report.feature_dimension,
                report.sample_rate,
                report.speaker_count,
                report.uses_f0
            );
            match report.index_vectors {
                Some(vectors) => println!("loaded retrieval index with {vectors} vectors"),
                None => println!("retrieval index disabled"),
            }
            Ok(())
        }
        Some("convert") => {
            let checkpoint = required(&mut arguments, "model.pth")?;
            let contentvec = required(&mut arguments, "hubert_base.pt")?;
            let index = optional_path(required(&mut arguments, "model.index or -")?);
            let input_audio = required(&mut arguments, "input audio")?;
            let output_audio = required(&mut arguments, "output.wav")?;
            let pitch_shift = arguments
                .next()
                .as_deref()
                .unwrap_or("0")
                .parse::<i8>()
                .map_err(|_| "pitch shift must be an integer semitone value")?;
            let device = parse_device(arguments.next().as_deref().unwrap_or("auto"))?;
            reject_extra(arguments)?;

            let mut engine = Engine::new();
            engine.set_config(EngineConfig {
                device,
                pitch_shift,
                retrieval_rate: if index.is_some() { 0.75 } else { 0.0 },
                ..EngineConfig::default()
            })?;
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(checkpoint),
                contentvec: Some(PathBuf::from(contentvec)),
                rmvpe: None,
                index,
            });
            let report = engine.start_offline(&OfflineJob {
                input_audio: PathBuf::from(input_audio),
                output_audio: PathBuf::from(output_audio),
            })?;
            println!(
                "converted {} samples at {} Hz in {:.2}s -> {}",
                report.samples,
                report.sample_rate,
                report.elapsed.as_secs_f32(),
                report.output_audio.display()
            );
            Ok(())
        }
        Some("status") => {
            reject_extra(arguments)?;
            println!("rvc-rs 0.4.0");
            println!("checkpoint/index: pthrs 0.2.0");
            println!("tensor backend: Candle 0.11.0");
            println!("native checkpoint loader: .pth -> pthrs -> Candle tensors");
            println!("native retrieval: FAISS IVF-Flat .index via pthrs");
            println!("desktop app: egui/eframe 0.36.1");
            println!("offline v2 path: native ContentVec + DSP F0 + RVC generator");
            Ok(())
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            print_usage();
            Ok(())
        }
        Some(command) => Err(format!("unknown command: {command}").into()),
    }
}

fn parse_device(value: &str) -> Result<ComputeDevice, String> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(ComputeDevice::Auto);
    }
    if value.eq_ignore_ascii_case("cpu") {
        return Ok(ComputeDevice::Cpu);
    }
    if let Some(index) = value.strip_prefix("cuda:") {
        return index
            .parse()
            .map(ComputeDevice::Cuda)
            .map_err(|_| format!("invalid CUDA device: {value}"));
    }
    if let Some(index) = value.strip_prefix("metal:") {
        return index
            .parse()
            .map(ComputeDevice::Metal)
            .map_err(|_| format!("invalid Metal device: {value}"));
    }
    Err(format!(
        "invalid device '{value}'; expected auto, cpu, cuda:N, or metal:N"
    ))
}

fn required(arguments: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    arguments
        .next()
        .ok_or_else(|| format!("missing argument: {name}"))
}

fn optional_path(value: String) -> Option<PathBuf> {
    (value != "-").then(|| PathBuf::from(value))
}

fn reject_extra(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    if let Some(argument) = arguments.next() {
        return Err(format!("unexpected argument: {argument}"));
    }
    Ok(())
}

fn print_usage() {
    println!("rvc-rs — Rust RVC file-conversion CLI");
    println!();
    println!("Usage:");
    println!("  rvc-rs status");
    println!("  rvc-rs doctor [auto|cpu|cuda:N|metal:N]");
    println!(concat!(
        "  rvc-rs validate-offline <model.pth> <model.index|-> ",
        "<input> <output.wav>"
    ));
    println!(concat!(
        "  rvc-rs prepare-native <model.pth> <model.index|-> ",
        "[auto|cpu|cuda:N|metal:N]"
    ));
    println!(concat!(
        "  rvc-rs convert <model.pth> <hubert_base.pt> <model.index|-> ",
        "<input> <output.wav> [pitch-semitones] [auto|cpu|cuda:N|metal:N]"
    ));
    println!();
    println!("The production direction is direct .pth/.index inference on Candle.");
}
