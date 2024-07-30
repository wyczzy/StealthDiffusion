Label = {0: 'ai',
         1: 'nature'}

# TODO: Find a better way for the prompt selection, whether random select a prompt from multiple choices
refined_Label = {}
for k, v in Label.items():
    if len(v.split(",")) == 1:
        select = v
    else:
        select = v.split(",")[0]
    refined_Label.update({k: select})
