"""Coswara dataset."""

import os
import pathlib
import tensorflow_datasets as tfds

_DESCRIPTION = """
Drawn by the information packed nature of sound signals, Project Coswara aims to evaluate 
effectiveness for COVID-19 diagnosis using sound samples. The idea is to create a huge dataset 
of breath, cough, and speech sound, drawn from healthy and COVID-19 positive individuals, 
from around the globe. Subsequently, the dataset will be analysed using signal processing 
and machine learning techniques for evaluating the effectiveness in automatic detection of 
respiratory ailments, including COVID-19.
"""

_CITATION = """
@article{Sharma_2020,
   title={Coswara — A Database of Breathing, Cough, and Voice Sounds for COVID-19 Diagnosis},
   url={http://dx.doi.org/10.21437/Interspeech.2020-2768},
   DOI={10.21437/interspeech.2020-2768},
   journal={Interspeech 2020},
   publisher={ISCA},
   author={Sharma, Neeraj and Krishnan, Prashant and Kumar, Rohit and Ramoji, Shreyas and Chetupalli, Srikanth Raj and R., Nirmala and Ghosh, Prasanta Kumar and Ganapathy, Sriram},
   year={2020},
   month={Oct}
}
"""

LABEL_MAP = {
  'healthy': 0, 
  'positive_moderate': 1, 
  'positive_mild': 2, 
  'positive_asymp': 3
}

SAMPLE_RATE = 48000


class CoswaraConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Coswara Dataset."""

  def __init__(self, audio_file, **kwargs):
    """Constructs a CoswaraConfig.
    Args:
      audio-file: string, name of the audio file to include in the dataset. One of
        'breathing-ddeep', 'breathing-shallow', 'cough-heavy' and 'cough-shallow'
      **kwargs: keyword arguments forwarded to super.
    """

    super(CoswaraConfig, self).__init__(
        name=audio_file,
        version=tfds.core.Version("1.0.0"),
        description=f'Coswara dataset containing only {audio_file}.wav files.',
        **kwargs,
    )
    self.audio_file = audio_file

class Coswara(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Coswara dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [
      CoswaraConfig(audio_file='cough-heavy'),
      CoswaraConfig(audio_file='cough-shallow'),
      CoswaraConfig(audio_file='breathing-deep'),
      CoswaraConfig(audio_file='breathing-shallow')
  ]

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'label': tfds.features.ClassLabel(names=['healthy', 'positive_moderate', 'positive_mild', 'positive_asymp']),
            'audio': tfds.features.Audio(file_format='wav', sample_rate=SAMPLE_RATE),
            'user_id': tfds.features.Text(),
        }),
        supervised_keys=('audio', 'label'),  
        homepage='https://coswara.iisc.ac.in/?locale=en-US',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    path = dl_manager.download_and_extract('https://onedrive.live.com/download?cid=AE89039BCDF893AE&resid=AE89039BCDF893AE%21689&authkey=AJ6DQfPEHuBSENY') 

    path = path / 'dataset'
    return {
        'train': self._generate_examples(path / 'train'),
        'test': self._generate_examples(path / 'test'),
        'validation': self._generate_examples(path / 'validation'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    for user in path.glob("*"):
      with open(user / 'label.txt', 'r') as label_file:
        label = label_file.readline().strip()
      yield user.name, {
          'label': LABEL_MAP[label],
          'audio':  user / f'{self.builder_config.audio_file}.wav',
          'user_id': user.name
      }
