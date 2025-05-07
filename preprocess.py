import neattext.functions as nfx

def clean_text(text):
    text = nfx.remove_userhandles(text)
    text = nfx.remove_hashtags(text)
    text = nfx.remove_emojis(text)
    text = nfx.remove_special_characters(text)
    text = nfx.remove_stopwords(text)
    return text
