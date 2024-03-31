from audio_separator.separator import Separator
import os
from scipy.io import wavfile

weight_uvr5_root = os.getenv("weight_uvr5_root")

uvr5_names = []
class UVRHANDLER:
    def __init__(self, output_dir='opt', model_file_dir=weight_uvr5_root):
        self.output_dir = output_dir
        self.model_file_dir = model_file_dir
        self.separator = Separator(output_dir=self.output_dir, model_file_dir=self.model_file_dir)

    def loadmodel(self, model_name, output_dir='opt/', model_file_dir=weight_uvr5_root):
        self.output_dir = output_dir
        self.model_file_dir = model_file_dir
        self.separator = Separator(output_dir=self.output_dir, model_file_dir=self.model_file_dir)
        self.separator.load_model(model_name)
        globname = model_name
        return f'Loaded {model_name}!'

    def deloadmodel(self):
        del self.separator
        self.separator = Separator(output_dir=self.output_dir, model_file_dir=self.model_file_dir)
        return f'Unloaded!'

    def uvr(self, audio, outputdir='opt'):
        if audio is not None:
            file_name = audio.name
            #open(f'TEMP/{file_name}', 'w').write(audio)
            print(file_name)
            #inst, voc = self.separator.separate('TEMP/audio.wav')
            inst, voc = self.separator.separate(file_name)
            return self.output_dir+inst, self.output_dir+voc

    def set_output_dir(self, output_dir):
        self.output_dir = output_dir
