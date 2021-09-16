# A Paper List for Speech Translation 

### :hugs: We are hiring interns and full-time employees researching on speech translation, please contact me at dongqianqian@bytedance.com

This is a paper list for speech translation. 

**Keyword:** *Speech Translation, Spoken Language Processing, Natural Language Processing*
* [Tutorials and Surveys](#tutorial)
* [Codebase](#codebase)
* [Dataset](#dataset)
* [Paper List](#paper_list)
    * [Pipeline ST](#pipeline_st)
    * [End-to-end ST](#end_to_end_st)
    * [End-to-end Streaming ST](#end_to_end_streaming_st)
    * [End-to-end NA ST](#end_to_end_na_st)
    * [End-to-end Multilingual ST](#multilingual_st)
    * [End-to-end S2ST](#end_to_end_s2st)
    * [End-to-end Zero-shot ST](#end_to_end_zero_shot_st)
    * [Multimodal MT](#multimodal_mt)
    * [Streaming MT](#streaming_mt)
* [Related Works](#related_works)
    * [Automated Audio Captioning](#automated_audio_captioning)
    * [Named Entity Recognition](#named_entity_recognition)
    * [Text Normalization](#text_normalization)
    * [Disfluency Detection](#disfluency_detection)
    * [Punctuation Prediction](#punctuation_prediction)
* [Workshop](#workshop)

<h1 id="tutorial">Tutorials and Surveys</h1>

- Jan Niehues. Spoken Language Translation, InterSpeech-2019, [[video]](https://www.youtube.com/watch?v=beB5L6rsb0I)
- Matthias Sperber and Matthias Paulik. Speech Translation and the End-to-End Promise:Taking Stock of Where We Are, ACL-2020 theme track, [[paper]](https://arxiv.org/pdf/2004.06358) 
- Umut Sulubacak, Ozan Caglayan, Stig-Arne Grönroos, Aku Rouhe, Desmond Elliott, Lucia Specia, and Jörg Tiedemann. Multimodal Machine Translation through Visuals and Speech, Machine Translation journal-2020 (Springer), [[paper]](https://link.springer.com/article/10.1007/s10590-020-09250-0)
- Jan Niehues, Elizabeth Salesky, Marco Turchi, Matteo Negri. Speech Translation Tutorial, EACL-2021, [[link]](https://st-tutorial.github.io/), [[slides]](https://2021.eacl.org/downloads/tutorials/End-to-end-ST.pdf)

<h1 id="codebase">Codebase</h1>

- ESPnet-ST: All-in-One Speech Translation Toolkit, ACL-2020 Demo, [[paper]](https://arxiv.org/pdf/2004.10234), [[code]](https://github.com/espnet/espnet)
- FAIRSEQ S2T: Fast Speech-to-Text Modeling with FAIRSEQ, AACL-2020 demo, [[paper]](https://arxiv.org/pdf/2010.05171.pdf), [[code]](https://github.com/pytorch/fairseq)
- NeurST: Neural Speech Translation Toolkit, Arxiv-2020, [[paper]](https://arxiv.org/abs/2012.10018), [[code]](https://github.com/bytedance/neurst)

<h1 id="dataset">Dataset</h1>

- Construction and Utilization of Bilingual Speech Corpus for Simultaneous Machine Interpretation Research, InterSpeech-2005,[[paper]](https://www.isca-speech.org/archive/archive_papers/interspeech_2005/i05_1585.pdf)
- Approach to Corpus-based Interpreting Studies: Developing EPIC (European Parliament Interpreting Corpus), MuTra-2005, [[paper]](http://www.euroconferences.info/proceedings/2005_Proceedings/2005_Bendazzoli_Sandrelli.pdf)
- Automatic Translation from Parallel Speech: Simultaneous Interpretation as MT Training Data, ASRU-2009, [[paper]](http://isl.anthropomatik.kit.edu/pdf/Paulik2009.pdf)
- The KIT Lecture Corpus for Speech Translation, LREC-2012, [[paper]](http://www.lrec-conf.org/proceedings/lrec2012/pdf/1121_Paper.pdf)
- Improved Speech-to-Text Translation with the Fisher and Callhome Spanish–English Speech Translation Corpus, IWSLT-2013, [[paper]](http://www.mt-archive.info/10/IWSLT-2013-Post.pdf)
- Collection of a Simultaneous Translation Corpus for Comparative Analysis, LREC-2014, [[paper]](https://ahcweb01.naist.jp/papers/conference/2014/201405_LREC_Shimizu_1/201405_LREC_Shimizu_1.paper.pdf)
- Microsoft Speech Language Translation (MSLT) Corpus: The IWSLT 2016 release for English, French and German, IWSLT-2016, [[paper]](https://workshop2016.iwslt.org/downloads/IWSLT_2016_paper_12.pdf) 
- The Microsoft Speech Language Translation (MSLT) Corpus for Chinese and Japanese: Conversational Test data for Machine Translation and Speech Recognition, Machine_Translation-2017, [[paper]](https://github.com/MicrosoftTranslator/MSLT-Corpus)
- Amharic-English Speech Translation in Tourism Domain, SCNLP-2017, [[paper]](https://www.aclweb.org/anthology/W17-4608)
- A Very Low Resource Language Speech Corpus for Computational Language Documentation Experiment, LREC-2018, [[paper]](https://arxiv.org/pdf/1710.03501.pdf)
- Augmenting Librispeech with French Translations: A Multimodal Corpus for Direct Speech Translation Evaluation, LREC-2018, [[paper]](https://arxiv.org/abs/1802.03142)
- A Small Griko-Italian Speech Translation Corpus, SLTU-2019, [[paper]](https://arxiv.org/pdf/1807.10740.pdf)
- MuST-C: a Multilingual Speech Translation Corpus, NAACL-2019, [[paper]](https://www.aclweb.org/anthology/N19-1202)
- MaSS: A Large and Clean Multilingual Corpus of Sentence-aligned Spoken Utterances Extracted from the Bible, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1907.12895v2.pdf)
- How2: A Large-scale Dataset for Multimodal Language Understanding, NIPS-2018, [[paper]](https://arxiv.org/pdf/1811.00347.pdf)
- LibriVoxDeEn: A Corpus for German-to-English Speech Translation and Speech Recognition, LREC-2020, [[paper]](https://arxiv.org/pdf/1910.07924.pdf)
- Clotho: An Audio Captioning Dataset, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1910.09387.pdf)
- Europarl-St: A Multilingual Corpus For Speech Translation Of Parliamentary Debates, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1911.03167.pdf)
- CoVoST: A Diverse Multilingual Speech-To-Text Translation Corpus, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2002.01320)
- MuST-Cinema: a Speech-to-Subtitles corpus, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2002.10829)
- CoVoST 2: A Massively Multilingual Speech-to-Text Translation Corpus, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2007.10310), [[code]](https://github.com/pytorch/fairseq/tree/master/examples/speech_to_text)
- The Multilingual TEDx Corpus for Speech Recognition and Translation, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2102.01757.pdf)
- mintzai-ST: Corpus and Baselines for Basque-Spanish Speech Translation，IberSPEECH-2021，[[paper]](https://www.isca-speech.org/archive/IberSPEECH_2021/pdfs/41.pdf)
- BSTC: A Large-Scale Chinese-English Speech Translation Dataset, Arixv-2021, [[paper]](https://arxiv.org/abs/2104.03575)
- MultiSubs: A Large-scale Multimodal and Multilingual Dataset, Arxiv-2021, [[paper]](https://arxiv.org/abs/2103.01910)
- Kosp2e: Korean Speech to English Translation Corpus, InterSpeech-2021, [[paper]](https://arxiv.org/abs/2107.02875)

<h1 id="paper_list">Paper List</h1>

<h2 id="pipeline_st">Pipeline ST</h2>

- Phonetically-Oriented Word Error Alignment for Speech Recognition Error Analysis in Speech Translation, ASRU-2015,[[paper]](https://ieeexplore.ieee.org/document/7404808)
- Learning a Translation Model from Word Lattices, InterSpeech-2016, [[paper]](https://people.eng.unimelb.edu.au/tcohn/papers/adams16is.pdf)
- Learning a Lexicon and Translation Model from Phoneme Lattices, EMNLP-2016, [[paper]](https://www.aclweb.org/anthology/D16-1263)
- Neural Lattice-to-Sequence Models for Uncertain Inputs, EMNLP-2017, [[paper]](https://www.aclweb.org/anthology/D17-1145)
- Using Spoken Word Posterior Features in Neural Machine Translation, IWSLT-2018, [[paper]](https://ahcweb01.naist.jp/papers/conference/2018/201810_IWSLT_kaho-os_1/201810_IWSLT_kaho-os_1.paper.pdf)
- Towards robust neural machine translation, ACL-2018, [[paper]](https://www.aclweb.org/anthology/P18-1163)
- Assessing the Tolerance of Neural Machine Translation Systems Against Speech Recognition Errors, InterSpeech-2019, [[paper]](https://arxiv.org/pdf/1904.10997.pdf)
- Lattice Transformer for Speech Translation, ACL-2019, [[paper]](https://arxiv.org/pdf/1906.05551.pdf)
- Self-Attentional Models for Lattice Inputs, ACL-2019, [[paper]](https://arxiv.org/pdf/1906.01617.pdf)
- Breaking the Data Barrier: Towards Robust Speech Translation via Adversarial Stability Training, IWSLT-2019, [[paper]](https://arxiv.org/pdf/1909.11430.pdf)
- Neural machine translation with acoustic embedding, ASRU-2019
- Machine Translation in Pronunciation Space, Arxiv-2020, [[paper]](https://arxiv.org/pdf/1911.00932.pdf)
- Diversity by Phonetics and its Application in Neural Machine Translation, AAAI-2020, [[paper]](https://arxiv.org/pdf/1911.04292.pdf)
- Robust Neural Machine Translation for Clean and Noisy Speech Transcripts, IWSLT-2019, [[paper]](https://zenodo.org/record/3524947#.XfiAu5MzbGJ)
- ELITR Non-Native Speech Translation at IWSLT 2020, IWSLT-2020, [[paper]](https://arxiv.org/pdf/2006.03331)
- Subtitles to Segmentation: Improving Low-Resource Speech-to-Text Translation Pipelines, CLSST@LREC 2020, [[paper]](https://arxiv.org/pdf/2010.09693.pdf)
- Cascaded Models With Cyclic Feedback For Direct Speech Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11153.pdf)
- Sentence Boundary Augmentation For Neural Machine Translation Robustness, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11132.pdf)
- A Technical Report: But Speech Translation Systems, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11593.pdf)
- Direct Segmentation Models for Streaming Speech Translation, EMNLP-2020, [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.206.pdf)
- Lost in Interpreting: Speech Translation from Source or Interpreter?, InterSpeech-2021, [[paper]](https://arxiv.org/abs/2106.09343)
- Is “moby dick” a Whale or a Bird? Named Entities and Terminology in Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2109.07439.pdf)

<h2 id="end_to_end_st">End-to-end ST</h2> 

- Towards Speech Translation of Non Written Languages, IEEE-2006, [[paper]](https://ieeexplore.ieee.org/document/4123402)
- Towards speech-to-text translation without speech recognition, EACL-2017, [[paper]](https://www.aclweb.org/anthology/E17-2076)
- Listen and Translate: A Proof of Concept for End-to-End Speech-to-Text Translation, NIPS-2016, [[paper]](https://arxiv.org/abs/1612.01744)
- An Attentional Model for Speech Translation Without Transcription, NAACL-2016, [[paper]](https://www.aclweb.org/anthology/N16-1109)
- An Unsupervised Probability Model for Speech-to-Translation Alignment of Low-Resource Languages, EMNLP-2016, [[paper]](https://arxiv.org/pdf/1609.08139.pdf)
- A Case Study on Using Speech-to-translation Alignments for Language Documentation, ComputEL-2017, [[paper]](https://www.aclweb.org/anthology/W17-0123)
- Spoken Term Discovery for Language Documentation Using Translations, SCNLP-2017, [[paper]](https://www.aclweb.org/anthology/W17-4607)
- Sequence-to-sequence Models Can Directly Translate Foreign Speech, InterSpeech-2017, [[paper]](https://arxiv.org/pdf/1703.08581.pdf)
- Structured-based Curriculum Learning for End-to-end English-Japanese Speech Translation, InterSpeech-2017, [[paper]](https://arxiv.org/pdf/1802.06003.pdf)
- End-to-End Speech Translation with the Transformer, IberSPEECH-2018, [[paper]](https://www.isca-speech.org/archive/IberSPEECH_2018/pdfs/IberS18_P1-9_Cross-Vila.pdf)
- Towards Fluent Translations from Disfluent Speech, SLT-2018, [[paper]](https://arxiv.org/abs/1811.03189)
- Low-resource Speech-to-text Translation, InterSpeech-2018, [[paper]](https://arxiv.org/abs/1803.09164)
- End-to-End Automatic Speech Translation of Audiobooks, ICASSP-2018, [[paper]](https://arxiv.org/abs/1802.04200)
- Tied Multitask Learning for Neural Speech Translation, NAACL-2018, [[paper]](https://arxiv.org/abs/1802.06655)
- Towards Unsupervised Speech to Text Translation, ICASSP-2019, [[paper]](https://arxiv.org/pdf/1811.01307.pdf)
- Leveraging Weakly Supervised Data to Improve End-to-End Speech-to-Text Translation, ICASSP-2019, [[paper]](https://arxiv.org/abs/1811.02050.pdf)
- Towards End-to-end Speech-to-text Translation with Two-pass Decoding, ICASSP-2019, [[paper]](https://ieeexplore.ieee.org/document/8682801)
- Attention-Passing Models for Robust and Data-Efficient End-to-End Speech Translation, TACL-2019, [[paper]](https://arxiv.org/abs/1904.07209)
- End-to-End Speech Translation with Knowledge Distillation, InterSpeech-2019, [[paper]](https://arxiv.org/pdf/1904.08075)
- Fluent Translations from Disfluent Speech in End-to-End Speech Translation, NAACL-2019, [[paper]](https://arxiv.org/pdf/1906.00556)
- Pre-Training On High-Resource Speech Recognition Improves Low-Resource Speech-To-Text Translation, NAACL-2019, [[[paper]](https://arxiv.org/pdf/1809.01431.pdf)
- Exploring Phoneme-Level Speech Representations for End-to-End Speech Translation, ACL-2019, [[paper]](https://arxiv.org/pdf/1906.01199)
- Leveraging Out-of-Task Data for End-to-End Automatic Speech Translation, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1909.06515.pdf)
- Bridging the Gap between Pre-Training and Fine-Tuning for End-to-End Speech Translation, AAAI-2020, [[paper]](https://arxiv.org/pdf/1909.07575.pdf)
- Adapting Transformer to End-to-end Spoken Language Translation, InterSpeech-2019, [[paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3045.pdf)
- Unsupervised phonetic and word level discovery for speech to speech translation for unwritten languages, InterSpeech-2019, [[paper]](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/3026.pdf)
- A comparative study on end-to-end speech to text translation, ASRU-2019, [[paper]](https://arxiv.org/pdf/1911.08870.pdf)
- Instance-Based Model Adaptation For Direct Speech Translation, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1910.10663.pdf)
- Analyzing Asr Pretraining For Low-Resource Speech-To-Text Translation, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1910.10762.pdf)
- ON-TRAC Consortium End-to-End Speech Translation Systems for the IWSLT 2019 Shared Task, IWSLT-2019, [[paper]](https://arxiv.org/pdf/1910.13689.pdf)
- Harnessing Indirect Training Data for End-to-End Automatic Speech Translation: Tricks of the Trade, IWSLT-2019, [[paper]](https://arxiv.org/pdf/1909.06515v2.pdf)
- Data Efficient Direct Speech-to-Text Translation with Modality Agnostic Meta-Learning, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1911.04283.pdf)
- Enhancing Transformer for End-to-end Speech-to-Text Translation, EAMT-2019, [[paper]](https://www.aclweb.org/anthology/W19-6603.pdf)
- On Using SpecAugment for End-to-End Speech Translation, IWSLT-2019, [[paper]](https://arxiv.org/pdf/1911.08876.pdf)
- Synchronous Speech Recognition and Speech-to-Text Translation with Interactive Decoding, AAAI-2020, [[paper]](https://arxiv.org/abs/1912.07240)
- From Speech-To-Speech Translation To Automatic Dubbing, Arxiv-2020, [[paper]](https://arxiv.org/abs/2001.06785)
- Skinaugment: Auto-Encoding Speaker Conversions For Automaticspeech Translation, ICASSP-2020, [[paper]](https://arxiv.org/pdf/2002.12231)
- Curriculum Pre-training for End-to-End Speech Translation, ACL-2020, [[paper]](https://arxiv.org/pdf/2004.10093)
- Jointly Trained Transformers models for Spoken Language Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2004.12111)
- Relative Positional Encoding for Speech Recognition and Direct Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2005.09940)
- Worse WER, but Better BLEU? Leveraging Word Embedding asIntermediate in Multitask End-to-End Speech Translation, ACL-2020, [[paper]](https://arxiv.org/abs/2005.10678)
- Phone Features Improve Speech Translation, ACL-2020, [[paper]](https://arxiv.org/pdf/2005.13681)
- Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2005.11185)
- End-to-End Speech-Translation with Knowledge Distillation: FBK@IWSLT2020, IWSLT2020, [[paper]](https://arxiv.org/pdf/2006.02965)
- Self-Training for End-to-End Speech Translation, INTERSPEECH2020 (submitted), [[paper]](https://arxiv.org/pdf/2006.02490)
- CSTNet: Contrastive Speech Translation Network for Self-Supervised Speech Representation Learning, INTERSPEECH2020 (submitted), [[paper]](https://arxiv.org/pdf/2006.02814)
- Is 42 the Answer to Everything in Subtitling-oriented Speech Translation?, IWSLT2020, [[paper]](https://arxiv.org/pdf/2006.01080)
- End-To-End Speech Translation With Self-Contained Vocabulary Manipulation, ICASSP2020
- End-to-End Speech Translation With Transcoding by Multi-Task Learning for Distant Language Pairs, TASLP-2020, [[paper]](https://ahcweb01.naist.jp/papers/journal/2020/202005_IEEE_TASLP_takatomo-k/202005_IEEE_TASLP_takatomo-k.paper.pdf)
- UWSpeech: Speech to Speech Translation for Unwritten Languages, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2006.07926)
- Gender in Danger? Evaluating Speech Translation Technology on the MuST-SHE Corpus, ACL-2020, [[paper]](https://arxiv.org/pdf/2006.05754)
- Improving Cross-Lingual Transfer Learning for End-to-End Speech Recognition with Speech Translation, INTERSPEECH2020 (submitted), [[paper]](https://arxiv.org/pdf/2006.05474)
- Self-Supervised Representations Improve End-to-End Speech Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2006.12124)
- Consistent Transcription and Translation of Speech, TACL-2020, [[paper]](https://arxiv.org/pdf/2007.12741)
- Contextualized Translation of Automatically Segmented Speech, INTERSPEECH-2020, [[paper]](https://arxiv.org/pdf/2008.02270)
- On Target Segmentation for Direct Speech Translation, AMTA-2020, [[paper]](https://arxiv.org/pdf/2009.04707.pdf)
- End-to-End Speech Translation with Adversarial Training, WAST-2020, [[paper]](https://www.aclweb.org/anthology/2020.autosimtrans-1.2.pdf)
- SDST: Successive Decoding for Speech-to-text Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2009.09737)
- TED: Triple Supervision Decouples End-to-end Speech-to-text Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2009.09704)
- Investigating Self-supervised Pre-training for End-to-end Speech Translation, ICML-2020 workshop, [[paper]](https://openreview.net/pdf?id=SR2L__h9q9p), [[code]](https://github.com/mhn226/espnet)
- Adaptive Feature Selection for End-to-End Speech Translation, EMNLP2020 Findings, [[paper]](https://arxiv.org/pdf/2010.08518.pdf), [[code]](https://github.com/bzhangGo/zero)
- A General Multi-Task Learning Framework To Leverage Text Data For Speech To Text Tasks, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11338.pdf)
- MAM: Masked Acoustic Modeling for End-to-End Speech-to-Text Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11445.pdf)
- Evaluating Gender Bias In Speech Translation, ICASSP-2021 (submitted), [[paper]](https://arxiv.org/pdf/2010.14465.pdf)
- Bridging the Modality Gap for Speech-to-Text Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.14920.pdf)
- Dual-decoder Transformer for Joint Automatic Speech Recognition and Multilingual Speech Translation, COLING-2020, [[paper]](https://arxiv.org/abs/2011.00747), [[code]](https://github.com/formiel/speech-translation)
- Effectively pretraining a speech translation decoder with Machine Translation data, EMNLP-2020, [[paper]](https://www.aclweb.org/anthology/2020.emnlp-main.644.pdf)
- Tight Integrated End-to-End Training for Cascaded Speech Translation, SLT-2021, [[paper]](https://arxiv.org/pdf/2011.12167)
- Breeding Gender-aware Direct Speech Translation Systems, COLING-2020, [[paper]](https://arxiv.org/pdf/2012.04955.pdf)
- On Knowledge Distillation for Direct Speech Translation, CLiC-IT-2020, [[paper]](https://arxiv.org/pdf/2012.04964.pdf)
- Streaming Models for Joint Speech Recognition and Translation, EACL-2021, [[paper]](https://arxiv.org/pdf/2101.09149)
- CTC-based Compression for Direct Speech Translation, EACL-2021, [[paper]](https://arxiv.org/pdf/2102.01578)
- Fused Acoustic and Text Encoding for Multimodal Bilingual Pretraining and Speech Translation, ICML-2021, [[paper]](https://arxiv.org/pdf/2102.05766.pdf)
- Source and Target Bidirectional Knowledge Distillation for End-to-end Speech Translation, NAACL-2021, [[paper]](https://arxiv.org/pdf/2104.06457.pdf)
- Large-Scale Self- and Semi-Supervised Learning for Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2104.06678.pdf)
- End-to-end Speech Translation via Cross-modal Progressive Training, InterSpeech2021-2021, [[paper]](https://arxiv.org/abs/2104.10380)
- Beyond Voice Activity Detection: Hybrid Audio Segmentation for Direct Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2104.11710)
- AlloST: Low-resource Speech Translation without Source Transcription, InterSpeech2021-2021, [[paper]](https://arxiv.org/abs/2105.00171)
- Learning Shared Semantic Space for Speech-to-Text Translation, ACL-2021 Findings, [[paper]](https://arxiv.org/abs/2105.03095)
- Stacked Acoustic-and-Textual Encoding: Integrating the Pre-trained Models into Speech Translation Encoders, ACL-2021, [[paper]](https://arxiv.org/abs/2105.05752)
- How to Split: the Effect of Word Segmentation on Gender Bias in Speech Translation, ACL-2021 Findings, [[paper]](https://arxiv.org/pdf/2105.13782)
- Cascade versus Direct Speech Translation: Do the Differences Still Make a Difference?, ACL-2021, [[paper]](https://arxiv.org/abs/2106.01045)
- Efficient Transformer for Direct Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2107.03069)
- Improving Speech Translation by Understanding and Learning from the Auxiliary Text Translation Task, ACL-2021, [[paper]](https://arxiv.org/abs/2107.05782)
- Beyond Sentence-Level End-to-End Speech Translation: Context Helps, ACL-2021, [[paper]](https://aclanthology.org/2021.acl-long.200.pdf)
- AdaST: Dynamically Adapting Encoder States in the Decoder for End-to-End Speech-to-Text Translation, ACL-2021, [[paper]](https://aclanthology.org/2021.findings-acl.224.pdf)
- Speechformer: Reducing Information Loss in Direct Speech Translation, EMNLP-2021, [[paper]](https://arxiv.org/pdf/2109.04574.pdf)

<h2 id="end_to_end_streaming_st">End-to-end Streaming ST</h2>

- Simuls2s: End-to-end Simultaneous Speech To Speech Translation, ICLR-2019(under review), [[paper]](https://openreview.net/pdf?id=Ske_56EYvS)
- ON-TRAC Consortium for End-to-End and Simultaneous SpeechTranslation Challenge Tasks at IWSLT 2020, IWSLT-2020, [[paper]](https://arxiv.org/pdf/2005.11861)
- SimulSpeech: End-to-End Simultaneous Speech to Text Translation, ACL-2020, [[paper]](https://www.aclweb.org/anthology/2020.acl-main.350.pdf)
- Streaming Simultaneous Speech Translation With Augmented Memory Transformer, ICASSP-2021(submitted), [[paper]](https://arxiv.org/pdf/2011.00033.pdf)
- SimulMT to SimulST: Adapting Simultaneous Text Translation to End-to-End Simultaneous Speech Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2011.02048.pdf)
- Simultaneous Speech-To-Speech Translation System With Neural Incremental Asr, Mt, And Tts, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2011.04845.pdf)
- An Empirical Study Of End-To-End Simultaneous Speech Translation Decoding Strategies, ICASSP 2021, [[paper]](https://arxiv.org/pdf/2103.03233.pdf)
- RealTranS: End-to-End Simultaneous Speech Translation with Convolutional Weighted-Shrinking Transformer, ACL-2021 Findings, [[paper]](https://arxiv.org/abs/2106.04833)
- Direct Simultaneous Speech-to-Text Translation Assisted by Synchronized Streaming ASR, ACL-2021 Findings, [[paper]](https://arxiv.org/abs/2106.06636)
- Simultaneous Speech Translation for Live Subtitling: from Delay to Display, Arxiv-2021, [[paper]](https://arxiv.org/abs/2107.08807)
- UniST: Unified End-to-end Model for Streaming and Non-streaming Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2109.07368.pdf)

<h2 id="end_to_end_na_st">End-to-end NA ST</h2>

- Orthros: Non-Autoregressive End-To-End Speech Translation With Dual-Decoder, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.13047.pdf)
- Investigating the Reordering Capability in CTC-based Non-Autoregressive End-to-End Speech Translation, ACL-2021 Findings, [[paper]](https://arxiv.org/abs/2105.04840)
- Non-autoregressive End-to-end Speech Translation with Parallel Autoregressive Rescoring, Arxiv-2021, [[paper]](https://arxiv.org/abs/2109.04411)

<h2 id="multilingual_st">End-to-end Multilingual ST</h2>

- Multilingual End-To-End Speech Translation, ASRU-2019, [[paper]](https://arxiv.org/pdf/1910.00254.pdf)
- One-To-Many Multilingual End-To-End Speech Translation, ASRU-2019, [[paper]](https://arxiv.org/pdf/1910.03320.pdf)
- Multilingual Speech Translation with Efficient Finetuning of Pretrained Models, ACL-2021, [[paper]](https://arxiv.org/abs/2010.12829)
- Lightweight Adapter Tuning for Multilingual Speech Translation, [[paper]](https://arxiv.org/abs/2106.01463)

<h2 id="end_to_end_s2st">End-to-end S2ST</h2>

- Direct speech-to-speech translation with a sequence-to-sequence model, InterSpeech-2019, [[paper]](https://arxiv.org/pdf/1904.06037)
- Speech-To-Speech Translation Between Untranscribed Unknown Languages, ASRU-2019, [[paper]](https://arxiv.org/pdf/1910.00795.pdf)
- Transformer-Based Direct Speech-To-Speech Translation With Transcoder, SLT-2021, [[paper]](https://ahcweb01.naist.jp/papers/conference/2021/202101_SLT_takatomo-k/202101_SLT_takatomo-k.paper.pdf)
- Direct Speech-To-Speech Translation With Discrete Units, Arxiv-2021, [[paper]](https://arxiv.org/abs/2107.05604)
- Translatotron 2: Robust Direct Speech-To-Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2107.08661)

<h2 id="end_to_end_zero_shot_st">End-to-end Zero-shot ST</h2>

- Zero-shot Speech Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2107.06010)


<h2 id="multimodal_mt">Multimodal MT</h2>

- Transformer-based Cascaded Multimodal Speech Translation, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1910.13215.pdf)
- Towards Multimodal Simultaneous Neural Machine Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2004.03180)
- Towards Automatic Face-to-Face Translation, Arxiv-2020, [[paper]](https://arxiv.org/abs/2003.00418), [[code]](https://github.com/Rudrabha/LipGAN)
- Keyframe Segmentation and Positional Encoding for Video-guided Machine Translation Challenge 2020, ALVR-2020, [[paper]](https://arxiv.org/pdf/2006.12799.pdf)
- DeepFuse: HKU’s Multimodal Machine Translation System for VMT’20, ALVR-2020, [[paper]](https://alvr-workshop.github.io/2020/challenge_papers/DeepFuse.pdf)
- Team RUC AI·M3 Technical Report at VMT Challenge 2020: Enhancing Neural Machine Translation with Multimodal Rewards, ALVR-2020, [[paper]](https://alvr-workshop.github.io/2020/challenge_papers/ruc_report.pdf)
- Exploiting Multimodal Reinforcement Learning for Simultaneous Machine Translation，EACL-2021,[[paper]](https://arxiv.org/pdf/2102.11387.pdf)
- Cross-lingual Visual Pre-training for Multimodal Machine Translation, EACL-2021, [[paper]](https://arxiv.org/abs/2101.10044)
- Generative Imagination Elevates Machine Translation, NAACL-2021, [[https://arxiv.org/abs/2009.09654]]
- Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding, AAAI-2021, [[paper]](https://arxiv.org/pdf/2101.05208)
- Improving Translation Robustness with Visual Cues and Error Correction, Arxiv-2021, [[paper]](https://arxiv.org/abs/2103.07352)
- Gumbel-Attention for Multi-modal Machine Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2103.08862)
- Good for Misconceived Reasons: An Empirical Revisiting on the Need for Visual Context in Multimodal Machine Translation, ACL-2021, [[paper]](https://arxiv.org/abs/2105.14462)

<h2 id="streaming_mt">Streaming MT</h2>

- Simultaneous translation of lectures and speeches, Machine Translation-2007, [[paper]](https://ccc.inaoep.mx/~villasen/bib/Simultaneous%20translation%20of%20lectures%20and%20speeches.pdf)
- Real-time incremental speech-to-speech translation of dialogs, NAACL-2012, [[paper]](https://www.aclweb.org/anthology/N12-1048)
- Incremental segmentation and decoding strategies for simultaneous translation, IJCNLP-2013, [[paper]](https://www.aclweb.org/anthology/I13-1141)
- Don't Until the Final Verb Wait: Reinforcement learning for simultaneous machine translation, EMNLP-2014, [[paper]](https://www.aclweb.org/anthology/D14-1140)
- Segmentation strategies for streaming speech translation, NAACL-2013, [[paper]](https://www.aclweb.org/anthology/N13-1023)
- Optimizing segmentation strategies for simultaneous speech translation, ACL-2014, [[paper]](https://www.aclweb.org/anthology/P14-2090)
- Syntax-based simultaneous translation through prediction of unseen syntactic constituents, ACL-IJCNLP-2015, [[paper]](https://www.aclweb.org/anthology/P15-1020)
- Simultaneous machine translation using deep reinforcement learning, ICML-2016, [[paper]](http://tx.technion.ac.il/~danielm/icml_workshop/4.pdf)
- Interpretese vs. translationese: The uniqueness of human strategies in simultaneous interpretation, NAACL-2016, [[paper]](https://www.aclweb.org/anthology/N16-1111)
- Can neural machine translation do simultaneous translation?, Arxiv-2016, [[paper]](https://arxiv.org/abs/1606.02012)
- Learning to translate in real-time with neural machine translation, EACL-2017, [[paper]](https://arxiv.org/pdf/1610.00388.pdf)
- Incremental Decoding and Training Methods for Simultaneous Translation in Neural Machine Translation, NAACL-2018, [[paper]](https://www.aclweb.org/anthology/N18-2079)
- Prediction Improves Simultaneous Neural Machine Translation, EMNLP-2018, [[paper]](https://www.aclweb.org/anthology/D18-1337)
- STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency using Prefix-to-Prefix Framework, ACL-2019, [[paper]](https://arxiv.org/abs/1810.08398)
- Simultaneous Translation with Flexible Policy via Restricted Imitation Learning, ACL-2019, [[paper]](https://arxiv.org/pdf/1906.01135.pdf)
- Monotonic Infinite Lookback Attention for Simultaneous Machine Translation, ACL-2019, [[paper]](https://www.aclweb.org/anthology/P19-1126)
- Thinking Slow about Latency Evaluation for Simultaneous Machine Translation, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1906.00048.pdf)
- DuTongChuan: Context-aware Translation Model for Simultaneous Interpreting, Arxiv-2019, [[paper]](https://arxiv.org/abs/1907.12984)
- Monotonic Multihead Attention, ICLR-2020(under review), [[paper]](https://openreview.net/pdf?id=Hyg96gBKPS)
- How To Do Simultaneous Translation Better With Consecutive Neural Machine Translation, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1911.03154.pdf)
- Simultaneous Neural Machine Translation using Connectionist Temporal Classification, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1911.11933.pdf)
- Re-Translation Strategies For Long Form, Simultaneous, Spoken Language Translation, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1912.03393.pdf) 
- Learning Coupled Policies for Simultaneous Machine Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2002.04306)
- Re-translation versus Streaming for Simultaneous Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2004.03643)
- Efficient Wait-k Models for Simultaneous Machine Translation, Arxiv-2020, [[paper]](https://arxiv.org/abs/2005.08595)
- Opportunistic Decoding with Timely Correction for Simultaneous Translation, ACL-2020, [[paper]](https://arxiv.org/pdf/2005.00675)
- Neural Simultaneous Speech Translation Using Alignment-Based Chunking, IWSLT2020, [[paper]](https://arxiv.org/pdf/2005.14489)
- Dynamic Masking for Improved Stability in Spoken Language Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2006.00249)
- Learn to Use Future Informationin Simultaneous Translation, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2007.05290)
- Presenting Simultaneous Translation in Limited Space, ITAT WAFNL 2020, [[paper]](https://arxiv.org/pdf/2009.09016)
- Fluent and Low-latency Simultaneous Speech-to-Speech Translation with Self-adaptive Training, EMNLP2020 Findings, [[paper]](https://arxiv.org/pdf/2010.10048.pdf)
- Improving Simultaneous Translation with Pseudo References, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2010.11247.pdf)
- Future-Guided Incremental Transformer for Simultaneous Translation, AAAI-2021, [[paper]](https://arxiv.org/pdf/2012.12465.pdf)
- Faster Re-translation Using Non-Autoregressive Model For Simultaneous Neural Machine Translation, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2012.14681)
- Learning Coupled Policies for Simultaneous Machine Translation using Imitation Learning, EACL-2021, [[paper]](https://arxiv.org/abs/2002.04306)
- Simultaneous Multi-Pivot Neural Machine Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2104.07410)
- Stream-level Latency Evaluation for Simultaneous Machine Translation, Arxiv-2021, [[paper]](https://arxiv.org/abs/2104.08817)
- Impact of Encoding and Segmentation Strategies on End-to-End Simultaneous Speech Translation, Interspeech 2021, [[paper]](https://arxiv.org/abs/2104.14470)
- Full-Sentence Models Perform Better in Simultaneous Translation Using the Information Enhanced Decoding Strategy, Arxiv-2021, [[paper]](https://arxiv.org/pdf/2105.01893)
- Universal Simultaneous Machine Translation with Mixture-of-Experts Wait-k Policy, EMNLP-2021, [[paper]](https://arxiv.org/pdf/2109.05238.pdf)

<h1 id="related_works">Related Works</h1>

<h2 id="automated_audio_captioning">Automated Audio Captioning</h2>

- Effects Of Word-Frequency Based Pre- Annd Post- Processings For Audio Captioning, DCASE-2020, [[paper]](https://arxiv.org/pdf/2009.11436.pdf)

<h2 id="named_entity_recognition">Named Entity Recognition</h2>

- End-to-end Named Entity Recognition from English Speech, INTERSPEECH2020(submitted), [[paper]](https://arxiv.org/pdf/2005.11184)

<h2 id="text_normalization">Text Normalization</h2>

- A Hybrid Text Normalization System Using Multi-Head Self-Attention For Mandarin, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1911.04128.pdf)
- A Unified Sequence-To-Sequence Front-End Model For Mandarin Text-To-Speech Synthesis, ICASSP-2020, [[paper]](https://arxiv.org/pdf/1911.04111.pdf)
- Naturalization of Text by the Insertion of Pauses and Filler Words, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2011.03713.pdf)

<h2 id="disfluency_detection">Disfluency Detection</h2>

- Semi-Supervised Disfluency Detection, COLING-2018, [[paper]](https://www.aclweb.org/anthology/C18-1299.pdf)
- Adapting Translation Models for Transcript Disfluency Detection, AAAI-2019, [[paper]](https://www.aaai.org/ojs/index.php/AAAI/article/view/4597)
- Giving Attention to the Unexpected:Using Prosody Innovations in Disfluency Detection, Arxiv-2019, [[paper]](https://arxiv.org/pdf/1904.04388)
- Multi-Task Self-Supervised Learning for Disfluency Detection, AAAI-2020, [[paper]](https://arxiv.org/abs/1908.05378)
- Improving Disfluency Detection by Self-Training a Self-Attentive Model, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2004.05323)
- Combining Self-Training and Self-Supervised Learning for Unsupervised Disfluency Detection, EMNLP-2020, [[paper]](https://arxiv.org/pdf/2010.15360.pdf), [[code]](https://github.com/scir-zywang/self-training-self-supervised-disfluency/)
- Auxiliary Sequence Labeling Tasks For Disfluency Detection, Arxiv-2020, [[paper]](https://arxiv.org/pdf/2011.04512.pdf)

<h2 id="punctuation_prediction">Punctuation Prediction</h2>

- Controllable Time-Delay Transformer for Real-Time Punctuation Prediction and Disfluency Detection, ICASSP-2020，[[paper]](https://arxiv.org/pdf/2003.01309)
- Punctuation Prediction in Spontaneous Conversations: Can We Mitigate ASR Errors with Retrofitted Word Embeddings, INTERSPEECH-2020 (submitted), [[paper]](https://arxiv.org/pdf/2004.05985)
- Multimodal Semi-supervised Learning Framework for Punctuation Prediction in Conversational Speech, INTERSPEECH-2020, [[paper]](https://arxiv.org/pdf/2008.00702)

<h1 id="workshop">Workshop</h1>

- IWSLT 2018, [[link]](https://workshop2018.iwslt.org/index.php), [[paper]](https://workshop2018.iwslt.org/downloads/Proceedings_IWSLT_2018.pdf)
- IWSLT 2019, [[link]](http://workshop2019.iwslt.org/), [[paper]](https://zenodo.org/communities/iwslt2019/search?page=1&size=20)
- IWSLT 2020, [[link]](http://iwslt.org/doku.php?id=start), [[paper]](https://www.aclweb.org/anthology/2020.iwslt-1.1.pdf)
- AutoSimTrans 2020, [[link]](https://autosimtrans.github.io/), [[paper]](https://www.aclweb.org/anthology/2020.autosimtrans-1.pdf)
- Self-supervision in Audio and Speech, ICML 2020, [[link]](https://icml.cc/virtual/2020/workshop/5732)
- IWSLT 2021, [[link]](https://iwslt.org/2021/), [[paper]](https://aclanthology.org/2021.iwslt-1.pdf)
- AutoSimTrans 2021, [[link]](https://autosimtrans.github.io/)
- NAACL同传Workshop：千言 - 机器同传, [[link]](https://aistudio.baidu.com/aistudio/competition/detail/62)

<h1 id="copyright">Copyright</h1>

By volunteers from Institute of Automation，Chinese Academy of Sciences & ByteDance AI Lab.
  
**Welcome to open an issue or make a pull request!**
