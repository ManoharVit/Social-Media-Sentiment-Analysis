import re

def replace_filenames_and_save_new(script_path, train_filename, dev_filename, test_filename, new_script_path, new_emotion):
    try:
        # Read the script
        with open(script_path, 'r') as file:
            script_content = file.read()


        # Define explicit replacements
        script_content = script_content.replace('training/EI-oc-En-anger-train.txt', train_filename)
        script_content = script_content.replace('development/2018-EI-oc-En-anger-dev.txt', dev_filename)
        script_content = script_content.replace('test-gold/2018-EI-oc-En-anger-test-gold.txt', test_filename)
        script_content = script_content.replace('generate_intensity_mapping("anger")', f'generate_intensity_mapping("{new_emotion}")')
        script_content = script_content.replace( old_emotion, new_emotion)
        
        # Write modified script as a new file
        with open(new_script_path, 'w') as file:
            file.write(script_content)

        print(f"New script saved to {new_script_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")


script_path = '/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis/Notebook/anger/ml_classification_anger.py'
train_filename = 'training/EI-oc-En-fear-train.txt'
dev_filename = 'development/2018-EI-oc-En-fear-dev.txt'
test_filename = 'test-gold/2018-EI-oc-En-fear-test-gold.txt'
old_emotion = 'anger'
new_emotion = 'fear'

new_script_path = '/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis/Notebook/fear/ml_classification_fear.py'

replace_filenames_and_save_new(script_path, train_filename, dev_filename, test_filename, new_script_path, new_emotion)
