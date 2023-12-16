import speech_recognition as sr
from utils.measure_time import MeasureTime


class SpeechToTxt:
    @staticmethod
    def _record():
        r = sr.Recognizer()
        r.dynamic_energy_threshold = True
        # r.energy_threshold = 50
        # r.dynamic_energy_threshold = False
        r.energy_threshold = 20000
        r.pause_threshold = 2
        r.dynamic_energy_adjustment_ratio = 1.3
        # mic = sr.Microphone()
        # desired_sample_rate = 48000

        # mic = sr.Microphone(sample_rate=desired_sample_rate)
        mic = sr.Microphone()
        audio = None
        with mic as source:
            try:
                # r.adjust_for_ambient_noise(source)
                print('listening...')
                audio = r.listen(source)
                MeasureTime.start_measure_function_time('google')
                with open("data/temp.wav", "wb+") as f:
                    f.write(audio.get_wav_data())
                    print("saved")
                print('audio captured. \n')

            except Exception as e:
                print(e)
            finally:
                return r, audio

    @staticmethod
    def _recognize(record, audio):
        text_out = '>> recognition fail <<'
        try:
            # print('sending to google...')
            text_out = record.recognize_google(audio, language="pl-PL", show_all=False)
            time_consumed = MeasureTime.stop_measure_function_time('google')
            print(f'time consumed google: {time_consumed}')
            # print('received: ' + text_out)
        except Exception as e:
            print(e)
        finally:
            return text_out

    def listen(self) -> [str, None]:
        rec, audio = self._record()
        text = self._recognize(rec, audio).lower()
        if text:
            print('[' + text + ']')
            return text

        print('>> no match <<')
        return None


if __name__ == '__main__':
    sp = SpeechToTxt()
    print(sp.listen())
