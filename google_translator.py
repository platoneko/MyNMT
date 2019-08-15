from googletrans import Translator


translator = Translator(service_urls=['translate.google.cn'])


def transform(src):
    internal = translator.translate(src, dest='fr', src='en').text
    tgt = translator.translate(internal, dest='en', src='fr').text
    return tgt


if __name__ == "__main__":
    src = "Yeah those are men who just had sex\nNo we 're not no we 're not no we 're not"
    print(transform(src))
