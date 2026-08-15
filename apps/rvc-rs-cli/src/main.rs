#![forbid(unsafe_code)]

use rvc_rs_core::ComputeDevice;
use rvc_rs_engine::{Engine, EngineConfig, ModelFiles, OfflineJob, QualityPreset};
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

            if !rvc_rs_engine::hubert_is_cached() {
                eprintln!(
                    "managed HuBERT is not cached; downloading and verifying it once (189.5 MB)..."
                );
            }

            let mut engine = Engine::new();
            engine.set_config(EngineConfig {
                device,
                retrieval_rate: if index.is_some() { 0.75 } else { 0.0 },
                ..EngineConfig::default()
            })?;
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(model),
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
            let index = optional_path(required(&mut arguments, "model.index or -")?);
            let input_audio = required(&mut arguments, "input audio")?;
            let output_audio = required(&mut arguments, "output.wav")?;
            let config = parse_convert_options(arguments.collect(), index.is_some())?;

            if !rvc_rs_engine::hubert_is_cached() {
                eprintln!(
                    "managed HuBERT is not cached; downloading and verifying it once (189.5 MB)..."
                );
            }

            let mut engine = Engine::new();
            engine.set_config(config)?;
            engine.set_model(ModelFiles {
                checkpoint: PathBuf::from(checkpoint),
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
            println!("rvc-rs {}", env!("CARGO_PKG_VERSION"));
            println!("checkpoint/index: pthrs 0.2.0");
            println!("tensor backend: Candle 0.11.0");
            println!("native checkpoint loader: .pth -> pthrs -> Candle tensors");
            println!("native retrieval: FAISS IVF-Flat .index via pthrs");
            println!("desktop app: egui/eframe 0.36.1");
            println!("offline v2 path: native ContentVec + DSP F0 + RVC generator");
            println!(
                "managed HuBERT: {}",
                if rvc_rs_engine::hubert_is_cached() {
                    "cached; integrity is verified before inference"
                } else {
                    "will download automatically on first conversion"
                }
            );
            Ok(())
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            print_usage();
            Ok(())
        }
        Some(command) => Err(format!("unknown command: {command}").into()),
    }
}

fn parse_convert_options(values: Vec<String>, has_index: bool) -> Result<EngineConfig, String> {
    let mut position = 0;
    let mut positional_pitch = None;
    let mut positional_device = None;
    if values.get(position).is_some_and(|value| !value.starts_with("--")) {
        positional_pitch = Some(parse_number::<i8>(&values[position], "pitch shift")?);
        position += 1;
    }
    if values.get(position).is_some_and(|value| !value.starts_with("--")) {
        positional_device = Some(parse_device(&values[position])?);
        position += 1;
    }

    let options = &values[position..];
    let mut preset = QualityPreset::Balanced;
    let mut cursor = 0;
    while cursor < options.len() {
        let name = &options[cursor];
        let value = options
            .get(cursor + 1)
            .ok_or_else(|| format!("missing value after {name}"))?;
        if name == "--preset" {
            preset = QualityPreset::parse(value).ok_or_else(|| {
                format!(
                    "invalid preset '{value}'; expected balanced, clean, singing, or identity"
                )
            })?;
        }
        cursor += 2;
    }

    let mut config = EngineConfig::default();
    preset.apply(&mut config);
    if let Some(pitch) = positional_pitch {
        config.pitch_shift = pitch;
    }
    if let Some(device) = positional_device {
        config.device = device;
    }

    cursor = 0;
    while cursor < options.len() {
        let name = &options[cursor];
        let value = &options[cursor + 1];
        match name.as_str() {
            "--preset" => {}
            "--pitch" => config.pitch_shift = parse_number(value, name)?,
            "--device" => config.device = parse_device(value)?,
            "--speaker" => config.speaker_id = parse_number(value, name)?,
            "--index-rate" => config.retrieval_rate = parse_number(value, name)?,
            "--index-k" => config.retrieval_neighbors = parse_number(value, name)?,
            "--index-nprobe" => config.retrieval_nprobe = parse_number(value, name)?,
            "--protect" => config.protect_rate = parse_number(value, name)?,
            "--noise-scale" => config.noise_scale = parse_number(value, name)?,
            "--f0-min" => config.f0_min_hz = parse_number(value, name)?,
            "--f0-max" => config.f0_max_hz = parse_number(value, name)?,
            "--f0-threshold" => config.f0_yin_threshold = parse_number(value, name)?,
            "--f0-filter-radius" => config.f0_filter_radius = parse_number(value, name)?,
            "--rms-mix" => config.rms_mix_rate = parse_number(value, name)?,
            "--gain-db" => config.output_gain_db = parse_number(value, name)?,
            _ => return Err(format!("unknown convert option: {name}")),
        }
        cursor += 2;
    }
    if !has_index {
        config.retrieval_rate = 0.0;
    }
    Ok(config)
}

fn parse_number<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| format!("invalid value for {name}: {value}"))
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
        "  rvc-rs convert <model.pth> <model.index|-> ",
        "<input> <output.wav> [pitch-semitones] [auto|cpu|cuda:N|metal:N] [options]"
    ));
    println!();
    println!("Conversion options (preset is applied before individual overrides):");
    println!("  --preset <balanced|clean|singing|identity>");
    println!("  --pitch <-24..24>              --device <auto|cpu|cuda:N|metal:N>");
    println!("  --speaker <id>                 --index-rate <0..1>");
    println!("  --index-k <1..32>              --index-nprobe <1..64>");
    println!("  --protect <0..0.5>             --noise-scale <0..1.5>");
    println!("  --f0-min <40..300>             --f0-max <300..1600>");
    println!("  --f0-threshold <0.05..0.40>    --f0-filter-radius <0..7>");
    println!("  --rms-mix <0..1>               --gain-db <-24..12>");
    println!();
    println!("The production direction is direct .pth/.index inference on Candle.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_positionals_and_new_overrides_can_be_combined() {
        let config = parse_convert_options(
            [
                "-2",
                "cpu",
                "--preset",
                "singing",
                "--index-rate",
                "0.45",
                "--noise-scale",
                "0.3",
            ]
            .into_iter()
            .map(|value| value.to_owned())
            .collect(),
            true,
        )
        .unwrap();
        assert_eq!(config.pitch_shift, -2);
        assert_eq!(config.device, ComputeDevice::Cpu);
        assert_eq!(config.f0_max_hz, 1_400.0);
        assert_eq!(config.retrieval_rate, 0.45);
        assert_eq!(config.noise_scale, 0.3);
    }

    #[test]
    fn dash_index_forces_retrieval_off() {
        let config = parse_convert_options(
            ["--preset", "identity"]
                .into_iter()
                .map(|value| value.to_owned())
                .collect(),
            false,
        )
        .unwrap();
        assert_eq!(config.retrieval_rate, 0.0);
    }
}
