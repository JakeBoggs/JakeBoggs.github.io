---
title: "Daily Yap: A Synthetically Generated Conversational Audio Dataset"
date: "2024-06-23"
draft: false
summary: "90-hour conversational audio dataset. Used GPT-4o to clean up Daily Dialog transcripts and XTTS v2 to synthesize speech with 8 different voices. Available on HuggingFace."
---
Training conversational audio models requires large, high-quality datasets, which are scarce at the time of writing.

There are many limitations with existing audio datasets:

1.  **Content Scope:** Most focus on assistant-user interactions, lacking the breadth of topics found in general human dialogues.
2.  **Audio-Text Alignment:** Datasets with precise alignment between high-quality audio and accurate transcriptions are uncommon.
3.  **Speaker Diversity:** The use of few speakers limits the generalizability of models trained on these datasets.
4.  **Scalability:** Human recording is resource-intensive, hindering the creation of large-scale datasets.

Daily Yap was created to overcome these challenges by synthetically generating conversational audio from an existing text-only dataset. This ensures a large diversity of topics and perfect annotation. Daily Dialog was selected as the foundation for this, with some modifications to enhance its suitability for audio synthesis.

The transcripts were initially filtered to remove conversations where any utterance was shorter than 10 characters. The remaining transcripts were then processed using GPT-4o via the OpenAI API with three main objectives:

1.  Correcting grammatical and spelling errors.
2.  Reformatting text for improved compatibility with text-to-speech (TTS) engines. This included expanding abbreviations (e.g., "Mr." to "Mister", ".com" to "dot com", "@" to "at") so the text represented spoken language more accurately.
3.  Extending shorter conversations. GPT-4o was prompted to plausibly continue the dialogue until it reached a total of 10 utterances, increasing the average length and complexity.

The prompt instructs GPT-4o to return the processed dialogue as a JSON object containing a list of strings, where each string represents one turn in the conversation (e.g., `{"segments": ["segment one", "segment two", ...]}`).

```python
completion = client.chat.completions.create(
    model='gpt-4o',
    response_format={ 'type': 'json_object' },
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant who responds only with JSON.'},
        {'role': 'user', 'content': 'Given the following dialog transcription, fix any formatting issues, grammar mistakes, or spelling mistakes if they exist. Replace any abbreviations like the ".? in ".com" or the "@" in an email with their corresponding text, such as "dot" or "at" so that the text reads the way it would be spoken. Also replace other common abbreviations like "Ms" and "Mr". Additionally, if you think it is plausible that the conversation could continue, generate some additional lines of dialog until there are 10 total. Reply with a JSON object in the following format: {"segments": ["segment one", "segment two"]}\n: ' + json.dumps(sample['dialog'])}
    ]
)
completion_data = json.loads(completion.choices[0].message.content)['segments']
```

To generate the audio cheaply, I searched for open-source TTS models that I could run locally. After evaluating several, including ChatTTS, Coqui TTS `xtts_v2` was selected for the perceived naturalness of its output.

```python
from TTS.api import TTS

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=True)
```

To introduce speaker variability, eight distinct voices (four male, four female) were synthesized using recorded samples as inputs to generate latent representations. For each conversation transcript, two distinct speaker voices were randomly selected from the available voice pool.

The generation process iterates through each utterance in the conversation's segments. Each turn is assigned to one of the two selected voices in an alternating fashion. The `tts.tts_to_file` function generates a temporary WAV file for the utterance using the appropriate speaker voice and language set to English.

To create the dual-channel output, two `AudioSegment` objects (`track_one`, `track_two`) are initialized. As each utterance's audio is generated, it is appended to the corresponding speaker's track. Simultaneously, silence of equivalent duration is appended to the *other* speaker's track using `AudioSegment.silent()`. This ensures both tracks remained synchronized, representing the back-and-forth nature of the conversation with silence during the other speaker's turn.

```python
from os import listdir
from random import choice
from pydub import AudioSegment

voices = listdir('voices')
# ... (dataset loading)

for (i, sample) in enumerate(dataset):
    # Select two distinct random voices
    v1 = 'voices/' + choice(voices)
    v2 = 'voices/' + choice(voices)
    while v1 == v2:
        v2 = 'voices/' + choice(voices)

    track_one = AudioSegment.empty()
    track_two = AudioSegment.empty()

    for (j, line) in enumerate(sample['conversation']):
        if j % 2 == 1: # Speaker 1
            tts.tts_to_file(text=line, file_path='current.wav', speaker_wav=v1, language='en')
            segment = AudioSegment.from_wav('current.wav')
            track_one += segment
            track_two += AudioSegment.silent(duration=len(segment))
        else: # Speaker 0
            tts.tts_to_file(text=line, file_path='current.wav', speaker_wav=v2, language='en')
            segment = AudioSegment.from_wav('current.wav')
            track_two += segment
            track_one += AudioSegment.silent(duration=len(segment))
    
    # Combine tracks and export
    result = AudioSegment.from_mono_audiosegments(
        track_one[:min(len(track_one), len(track_two))],
        track_two[:min(len(track_one), len(track_two))]
    )
    result.export('audio/' + str(i) + '.mp3', format='mp3')
```

Finally, the two mono audio segments are combined into a single stereo audio file using `AudioSegment.from_mono_audiosegments`. The resulting segment is truncated to the length of the shorter track to handle any minor duration discrepancies and then exported as an MP3 file.

The resulting Daily Yap dataset contains 9,758 samples, totaling approximately 90 hours of audio. Each sample consists of a JSON-formatted transcript and a corresponding dual-channel WAV audio file. The average sample duration is 33 seconds.

The dataset is available on HuggingFace: [https://huggingface.co/datasets/jakeBoggs/DailyYap](https://huggingface.co/datasets/jakeBoggs/DailyYap)

If any researchers want to cite this in a paper, I would be both honored and amused. Seeing "Daily Yap" in a works cited section would give me a good laugh.