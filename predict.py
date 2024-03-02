from TTS.utils.synthesizer import Synthesizer
from cog import BasePredictor, Path, Input, File
import io

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.synthesizer = Synthesizer(
                tts_checkpoint="model.pth",
                tts_config_path="config.json",
                vocoder_checkpoint="vocoder.pth",
                vocoder_config="vocoder_config.json",
                use_cuda=False
                )
    def predict(self, text: str = Input(description="Text to synthesize")) -> File:
        """Run a single prediction on the model"""
        wavs = self.synthesizer.tts(text)
        out = io.BytesIO()
        self.synthesizer.save_wav(wavs, out)

        return out
