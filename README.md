# TextSummarizationITAcademy

Код, использованный при подготовке лекции https://www.youtube.com/watch?v=HCaPhQ2Ub0E

Зависимости: pytorch и перечисленные в requirements.txt

Данные: 
- gazeta.ru https://github.com/IlyaGusev/gazeta
- lenta.ru https://github.com/yutkin/Lenta.Ru-News-Dataset


**lenta_simple_extractive.ipynb** - воспроизведение continuous LexRank

**gazeta_mbart.ipynb** - использование претренированной модели mBart (https://huggingface.co/IlyaGusev/mbart_ru_sum_gazeta)

**Gazeta_gpt_summarizer_training.ipynb** - попытка обучить модель на основе rugpt-3

**Gazeta_gpt_evaluation.ipynb** - оценка модели, обучаемой в предыдущем notebook
