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
            let input_audio = required(&mut arguments, "input audio")?;
            let output_audio = required(&mut arguments, "output.wav")?;
            let index = arguments.next().map(PathBuf::from);
            reject_extra(arguments)?;

            let mut engine = Engine::new();
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(checkpoint),
                index,
            });
            engine.validate_offline(&OfflineJob {
                input_audio: PathBuf::from(input_audio),
                output_audio: PathBuf::from(output_audio),
            })?;
            println!("offline job paths and configuration are valid");
            println!("generator execution remains gated until reference parity passes");
            Ok(())
        }
        Some("status") => {
            reject_extra(arguments)?;
            println!("rvc-rs 0.1.0 scaffold");
            println!("checkpoint/index: pthrs 0.2.0");
            println!("tensor backend: Candle 0.11.0");
            println!("desktop app: egui/eframe 0.36.1");
            println!("next milestone: v2/40k/F0 offline generator parity");
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

fn reject_extra(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    if let Some(argument) = arguments.next() {
        return Err(format!("unexpected argument: {argument}"));
    }
    Ok(())
}

fn print_usage() {
    println!("rvc-rs — native Rust RVC development CLI");
    println!();
    println!("Usage:");
    println!("  rvc-rs status");
    println!("  rvc-rs doctor [auto|cpu|cuda:N|metal:N]");
    println!("  rvc-rs validate-offline <model.pth> <input> <output.wav> [model.index]");
    println!();
    println!("Voice conversion will unlock after generator reference parity passes.");
}

