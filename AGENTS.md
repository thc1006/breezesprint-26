# Working on BreezeSprint26

Read README.md, src/breezesprint26/model.json and docs/VALIDATION.md before changes.
Model is fixed to MediaTek-Research/Breeze-ASR-26 at 26f44dc0e75a5044d81c27f8e9679a95619686b6; never substitute Turbo,
ASR-25/26 or a CTranslate2 conversion silently. Keep exactly two notebook cells and
upload-only input / timestamped TXT-only automatic output. Never hide errors,
remove words/repeats, shorten token budgets or call an external ASR API.

Core equations are unchanged from the source WhisperSprint engine. New model
profiles require real audio tests; CPU tests are not TPU certification. The fixed
zh token is an app decision, not an official benchmark claim. ASR-26 renders Taigi speech in Mandarin Chinese characters. It does not promise native Taigi orthography, Tâi-lô, or phonetic verbatim transcription. It is not a universal upgrade over ASR-25 for Mandarin.

Run pytest with JAX_PLATFORMS=cpu for local tests, then rebuild notebook and run
check_release.py. Actual model weights are not included. Do not commit caches,
recordings, generated transcripts, private paths or credentials. New test evidence
must distinguish synthetic, mocked and physical runs. Keep notices/license intact.

Tokenizer regressions: run both the offline suite and the real metadata-only
`tools/check_model_assets.py` check before claiming publisher-asset compatibility.
Do not confuse a synthetic full-ID-range fixture with the full pretrained vocab.
Never replace missing special tokens with guessed numeric constants. Rebuild the
standalone notebook after any source change.
