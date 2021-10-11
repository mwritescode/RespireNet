"""Coswara dataset."""

import random
import librosa
import tensorflow_datasets as tfds

from pydub import AudioSegment
from pydub.utils import make_chunks

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
POSSIBLE_SKIPS = range(1, 6)
CHUNK_LENGHT_MS = 3000 


class CoswaraConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Coswara Dataset."""

  def __init__(self, audio_file, skip, mixup=False, **kwargs):
    """Constructs a CoswaraConfig.
    Args:
      audio-file: string, name of the audio file to include in the dataset. One of
        'breathing-ddeep', 'breathing-shallow', 'cough-heavy' and 'cough-shallow'
      **kwargs: keyword arguments forwarded to super.
    """
    mixup_string = 'mixup' if mixup else ''

    super(CoswaraConfig, self).__init__(
        name=audio_file + '-skip{}-{}'.format(skip, mixup_string),
        version=tfds.core.Version("1.0.0"),
        description=f'Coswara dataset containing only {audio_file}.wav files.',
        **kwargs,
    )
    self.audio_file = audio_file
    self.skip = skip
    self.mixup = mixup

class Coswara(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Coswara dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  BUILDER_CONFIGS = [CoswaraConfig(audio_file='cough-heavy', skip=i) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='cough-heavy', skip=i, mixup=True) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='cough-shallow', skip=i) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='cough-shallow', skip=i, mixup=True) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='breathing-deep', skip=i) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='breathing-deep', skip=i, mixup=True) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='breathing-shallow', skip=i) for i in POSSIBLE_SKIPS] \
    + [CoswaraConfig(audio_file='breathing-shallow', skip=i, mixup=True) for i in POSSIBLE_SKIPS]

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
    gen_path = self._generate_mixup_examples(path)
    if path.name == 'train':
      file_list = list(path.glob("*")) + list(gen_path.glob("*"))
    else:
      file_list = path.glob("*")
    for user in file_list:
      if user.name != 'new':
        with open(user / 'label.txt', 'r') as label_file:
          label = label_file.readline().strip()
        audiofile = user / f'{self.builder_config.audio_file}.wav'
        duration = AudioSegment.from_file(audiofile).duration_seconds
        if duration >= self.builder_config.skip:
          yield user.name, {
              'label': LABEL_MAP[label],
              'audio':  user / f'{self.builder_config.audio_file}.wav',
              'user_id': user.name
          }  
  
  def _generate_mixup_examples(self, path):
      save_generated = path.parent / 'new'
      if path.name == 'train' and self.builder_config.mixup and not save_generated.exists():
        print('Generating new examples for the rare classes...')
        save_generated.mkdir(parents=True, exist_ok=True)
        positive_users = {
          'positive_mild': [],
          'positive_moderate': [],
          'positive_asymp': []
        }
        for user in path.glob("*"):
          if user.name != 'new':
            with open(user / 'label.txt', 'r') as label_file:
              label = label_file.readline().strip()
              if 'positive' in label:
                positive_users[label].append(user)
        
        for label, users in positive_users.items():
          num_repetitions = len(users)
          for _ in range(num_repetitions):
            user_1, chunks_1 = self._get_nonempty_chunk_list(users)
            user_2, chunks_2 = self._get_nonempty_chunk_list(users)
            new_user_path = save_generated / f'{user_1.name}-{user_2.name}'
            new_user_path.mkdir(parents=True, exist_ok=True)
            
            new_audio = random.choice(chunks_1) + random.choice(chunks_2)
            new_audio.export(new_user_path / f'{self.builder_config.audio_file}.wav', format='wav')
            with open(new_user_path / 'label.txt', 'w') as labelfile:
              labelfile.write(label)

      return save_generated

  def _get_nonempty_chunk_list(self, users):
    chunks = []
    while len(chunks) == 0:
      user = random.choice(users)
      audio = AudioSegment.from_file(user / f'{self.builder_config.audio_file}.wav' , "wav") 
      chunks = make_chunks(audio, CHUNK_LENGHT_MS)
    return user, chunks
          
