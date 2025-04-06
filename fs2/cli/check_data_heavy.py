import torch
import torchaudio
from clipdetect import detect_clipping
from tqdm import tqdm


def check_datapoint(
    item,
    preprocessor,
    evaluation_model,
    word_seg_token=" ",
    heavy_clip_detection=False,
    heavy_objective_evaluation=False,
):
    # speaking rate (words/second, float, scatterplot or bar chart)
    # speaking rate (characters/second, float, scatterplot or bar chart)
    # articulation level (mean energy/speaking rate)
    # unrecognized symbols (bool, list)
    # duration (float, box plot)
    # clipping (float, box plot)
    # silence % (float, box plot)
    data_point = {k: v for k, v in item.items()}
    characters = item.get("characters")
    character_tokens = item.get("character_tokens")
    phones = item.get("phones")
    phone_tokens = item.get("phone_tokens")
    assert (
        characters or phones
    ), "Sorry, your data does not have characters or phones available in the filelist, so we can't check the data."
    if character_tokens is None and phone_tokens is None:
        character_tokens, phone_tokens, _ = preprocessor.process_text(
            item, preprocessor.text_processor, use_pfs=False, encode_as_string=True
        )
    default_text = phones if phones is not None else characters
    n_words = len(default_text.split(word_seg_token))
    n_chars = len(character_tokens.split("/")) if character_tokens is not None else None
    n_phones = len(phone_tokens.split("/")) if phone_tokens is not None else None
    audio, sr = torchaudio.load(
        str(
            preprocessor.create_path(
                item, "audio", f"audio-{preprocessor.input_sampling_rate}.wav"
            )
        )
    )

    if heavy_objective_evaluation:
        # use objective metrics from https://pytorch.org/audio/main/tutorials/squim_tutorial.html
        if sr != 16000:
            model_audio = torchaudio.functional.resample(audio, sr, 16000)
        if len(model_audio.size()) == 1:  # must include channel
            model_audio = model_audio.unsqueeze(0)
        stoi_hyp, pesq_hyp, si_sdr_hyp = evaluation_model(model_audio)
        data_point["stoi"] = float(stoi_hyp[0])
        data_point["pesq"] = float(pesq_hyp[0])
        data_point["si_sdr"] = float(si_sdr_hyp[0])

    assert (
        len(audio.size()) == 1 or audio.size(0) == 1
    ), f"Audio has {audio.size(0)} channels, but should be mono"
    audio = audio.squeeze()

    if heavy_clip_detection:
        _, total_clipping = detect_clipping(audio)
    else:
        # this isn't a great way of detecting clipping,
        # but it's a lot faster than clipdetect
        audio_max = audio.max()
        audio_min = audio.min()
        total_clipping = (
            audio[audio >= audio_max].size(0) + audio[audio <= audio_min].size(0) - 2
        )
    pitch = torch.load(
        preprocessor.create_path(item, "pitch", "pitch.pt"), weights_only=True
    )
    energy = torch.load(
        preprocessor.create_path(item, "energy", "energy.pt"), weights_only=True
    )
    audio_length_s = len(audio) / preprocessor.input_sampling_rate
    data_point["total_clipped_samples"] = total_clipping
    data_point["pitch_min"] = float(pitch.min())
    data_point["pitch_max"] = float(pitch.max())
    data_point["pitch_mean"] = float(pitch.mean())
    data_point["pitch_std"] = float(pitch.std())
    data_point["energy_min"] = float(energy.min())
    data_point["energy_max"] = float(energy.max())
    data_point["energy_mean"] = float(energy.mean())
    data_point["energy_std"] = float(energy.std())
    data_point["duration"] = audio_length_s
    data_point["speaking_rate_words_per_second"] = n_words / audio_length_s
    if n_chars is not None:
        data_point["speaking_rate_characters_per_second"] = n_chars / audio_length_s
        data_point["n_chars"] = n_chars
    if n_phones is not None:
        data_point["speaking_rate_phones_per_second"] = n_phones / audio_length_s
        data_point["n_phones"] = n_phones
    data_point["n_missing_symbols"] = len(
        preprocessor.text_processor.get_missing_symbols(default_text)
    )
    data_point["n_words"] = n_words
    return data_point


def check_data_from_filelist(
    preprocessor,
    filelist,
    word_seg_token=" ",
    heavy_clip_detection=False,
    heavy_objective_evaluation=False,
):
    data = []
    if heavy_objective_evaluation:
        model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
    else:
        model = None
    for item in tqdm(filelist, desc="Checking Data"):
        data_point = check_datapoint(
            item,
            preprocessor,
            model,
            word_seg_token,
            heavy_clip_detection,
            heavy_objective_evaluation,
        )
        data.append(data_point)
    return data
