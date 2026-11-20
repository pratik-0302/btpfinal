"""What every displayed number means, and the evaluation results reported by the v4 notebook.

Single source of truth: the frontend shows these texts (served by GET /api/model).
All evaluation numbers are copied from the saved outputs of notebook cells C4 and E1.
Nothing here is recomputed or estimated.
"""
from __future__ import annotations

SCORE_DEFINITIONS = {
    "verdict": {
        "label": "Verdict",
        "meaning": "REAL when the fake score is 35 or below, FAKE when it is 65 or above, UNCERTAIN in between. "
                   "Forced to UNCERTAIN (fake score 50) when the clip has less than 0.6 s of speech or a face in "
                   "fewer than half of the frames.",
        "not": "",
    },
    "fake_score": {
        "label": "Fake score (0–100)",
        "meaning": "The larger of the two check scores below. A display scale on which 35 and 65 are the decision "
                   "thresholds chosen by grid search on validation data. Real score = 100 − fake score.",
        "not": "Not a probability, a confidence level or an accuracy.",
    },
    "sync_score": {
        "label": "Sync-check score",
        "meaning": "Score of the classifier that asks whether the audio belongs to these lips (catches swapped, dubbed "
                   "or delayed audio), on the same 0–100 display scale.",
        "not": "Not a probability.",
    },
    "deepfake_score": {
        "label": "Deepfake-check score",
        "meaning": "Score of the classifier that asks whether lips that do follow the speech move the way lip-sync "
                   "deepfakes do, on the same 0–100 display scale.",
        "not": "Not a probability.",
    },
    "raw_p": {
        "label": "Raw classifier output",
        "meaning": "Output of each logistic-regression classifier (0–1). The classifiers were trained with balanced "
                   "class weights; the deepfake output is the average over 6-second segments in logit space.",
        "not": "Not a real-world probability that the video is fake.",
    },
    "sync_z": {
        "label": "Sync strength (z)",
        "meaning": "How much higher the lip–speech agreement is at the best audio offset than at offsets at least "
                   "320 ms away, in standard errors (corrected for correlation between neighbouring frames). "
                   "Higher = the lips clearly follow this audio.",
        "not": "Not a probability.",
    },
    "matched_windows": {
        "label": "Matched speech windows",
        "meaning": "Share of 2-second speech windows (1 s apart) in which the lips clearly match the speech (window z > 2).",
        "not": "Not an accuracy.",
    },
    "best_offset_ms": {
        "label": "Audio offset",
        "meaning": "The audio shift (searched within ±240 ms) at which lips and speech agree best. Positive = the audio "
                   "comes later than the lip movement.",
        "not": "",
    },
    "agreement": {
        "label": "Lip–speech agreement",
        "meaning": "Cosine similarity (−1 to 1) between AV-HuBERT's lip embedding and speech embedding (final layer) "
                   "at the best audio offset, for every 40 ms frame. Window colours: green = lips match the speech "
                   "(z > 2), red = no match (z < 0.5), grey = unclear.",
        "not": "Not a per-frame fake probability.",
    },
    "segment_p": {
        "label": "Deepfake check per segment",
        "meaning": "Raw deepfake-classifier output for each 6-second segment (the first and last 4 frames are not used).",
        "not": "Not a localisation of manipulated frames.",
    },
    "face_detected_frac": {
        "label": "Face found",
        "meaning": "Share of the checked frames (every 2nd frame) in which a face was detected with probability ≥ 0.90.",
        "not": "Not an image-quality score.",
    },
    "mouth_width_px": {
        "label": "Mouth size",
        "meaning": "Median distance between the mouth corners in pixels, after the frame is scaled to at most 640 px.",
        "not": "",
    },
}

NEW_DISPLAYS = {
    "waveform": "New display calculation (not used by the model): loudness envelope of the 16 kHz mono audio the model received.",
    "audio_features": "New display calculation: the model's own audio input features — 26-band log-mel filterbank "
                      "energies (python_speech_features.logfbank), averaged to 25 frames per second.",
    "sync_matrix": "New display of existing data: lip–speech cosine similarity for every frame and every audio offset "
                   "from −600 ms to +600 ms (final AV-HuBERT layer). The detector computes this matrix; the notebook does not plot it.",
    "offset_drift": "New use of an existing function (window_offsets): best audio offset in 1-second windows, 0.5 s apart. "
                    "A steady line means steady timing; drift means the audio slides against the lips.",
}

# Notebook cell C4 / E1 outputs. "n" = number of videos in that set.
EVALUATION = {
    "source": "Saved outputs of notebook cells C4 and E1 (LipSync_Deepfake_Detector_v4). Test speakers were never used in training.",
    "headline_sets": [
        {"set": "LipSyncTIMIT test: real vs lip-sync fake", "n_real": 96, "n_fake": 1440,
         "auc": 0.992, "auc_ci": "0.987–0.996", "eer_pct": 6.771, "forced_choice_accuracy_pct": 94.857,
         "balanced_accuracy_pct": 94.340, "tpr_at_fpr1_pct": 89.931},
        {"set": "LipSyncTIMIT test: balanced subset", "n_real": 96, "n_fake": 96,
         "auc": 0.989, "auc_ci": "0.979–0.996", "eer_pct": 8.333, "forced_choice_accuracy_pct": 92.708,
         "balanced_accuracy_pct": 92.708, "tpr_at_fpr1_pct": 85.417},
        {"set": "LipSyncTIMIT test: camera/microphone varied (balanced)", "n_real": 96, "n_fake": 95,
         "auc": 0.925, "auc_ci": "0.878–0.964", "eer_pct": 10.471, "forced_choice_accuracy_pct": 89.529,
         "balanced_accuracy_pct": 89.550, "tpr_at_fpr1_pct": 0.0},
        {"set": "LipSyncTIMIT test: real vs out-of-sync copies", "n_real": 96, "n_fake": 191,
         "auc": 0.999, "auc_ci": "0.998–1.000", "eer_pct": 1.827, "forced_choice_accuracy_pct": 97.909,
         "balanced_accuracy_pct": 96.875, "tpr_at_fpr1_pct": 98.429},
    ],
    "verdict_breakdown": [
        {"set": "LipSyncTIMIT test real (unseen speakers)", "n": 96, "truth": "REAL", "real_pct": 90.6, "uncertain_pct": 9.4, "fake_pct": 0.0},
        {"set": "LipSyncTIMIT test lip-sync fakes", "n": 1440, "truth": "FAKE", "real_pct": 1.9, "uncertain_pct": 9.0, "fake_pct": 89.1},
        {"set": "LipSyncTIMIT test real, camera/mic varied", "n": 96, "truth": "REAL", "real_pct": 70.8, "uncertain_pct": 20.8, "fake_pct": 8.3},
        {"set": "LipSyncTIMIT test fakes, camera/mic varied", "n": 95, "truth": "FAKE", "real_pct": 4.2, "uncertain_pct": 8.4, "fake_pct": 87.4},
        {"set": "LipSyncTIMIT test made out-of-sync", "n": 191, "truth": "FAKE", "real_pct": 0.0, "uncertain_pct": 2.1, "fake_pct": 97.9},
        {"set": "Developer's own phone clips, genuine (leave-people-out)", "n": 29, "truth": "REAL", "real_pct": 93.1, "uncertain_pct": 6.9, "fake_pct": 0.0},
        {"set": "Own clips with another person's speech (leave-people-out)", "n": 794, "truth": "FAKE", "real_pct": 0.0, "uncertain_pct": 0.6, "fake_pct": 99.4},
        {"set": "Own clips with delayed audio (leave-people-out)", "n": 58, "truth": "FAKE", "real_pct": 0.0, "uncertain_pct": 1.7, "fake_pct": 98.3},
        {"set": "Own clips with 2–3.5 s of speech replaced (leave-people-out)", "n": 58, "truth": "FAKE", "real_pct": 19.0, "uncertain_pct": 19.0, "fake_pct": 62.1},
    ],
    "by_generator": [
        {"group": "Diff2Lip", "n_fake": 288, "auc": 0.986, "fake_pct": 83.0},
        {"group": "IP_LAP", "n_fake": 288, "auc": 0.998, "fake_pct": 95.8},
        {"group": "Video_Retalking", "n_fake": 288, "auc": 0.990, "fake_pct": 85.4},
        {"group": "Wav2Lip", "n_fake": 288, "auc": 0.993, "fake_pct": 91.7},
        {"group": "Wav2Lip_GAN", "n_fake": 288, "auc": 0.994, "fake_pct": 89.6},
        {"group": "fakes keeping the original audio", "n_fake": 480, "auc": 0.979, "fake_pct": 72.5},
        {"group": "fakes with LRS2 audio", "n_fake": 480, "auc": 0.999, "fake_pct": 97.5},
        {"group": "fakes with synthetic audio", "n_fake": 480, "auc": 0.998, "fake_pct": 97.3},
    ],
    "notes": [
        "The 29 genuine phone clips: 27 were called REAL and 2 UNCERTAIN (none FAKE). This measures false alarms on real "
        "videos only, on a small sample from 22 people, and the thresholds were partly tuned on these same scores. "
        "It is not an overall accuracy.",
        "No lip-sync deepfakes made from the developer's own recordings were tested.",
        "Forced-choice accuracy counts a video as fake when its fake score is 50 or more; the app itself also has an "
        "UNCERTAIN band (35–65).",
    ],
}

LIMITATIONS = [
    "Only lip–speech agreement is analysed: the detector never looks at the pixels or the sound itself, so face swaps "
    "that keep the original, matching audio are not its target.",
    "Only the first 30 seconds of a video are analysed.",
    "Works best with one clearly visible speaking face; with several faces the largest / most persistent one is used.",
    "Fakes that keep the original audio and partial audio replacements are detected less reliably (see the results).",
    "Trained mainly on LipSyncTIMIT; behaviour on very different footage can differ.",
]
