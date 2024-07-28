import re

def replace_filenames_and_save_new(script_path, train_filename, dev_filename, test_filename, new_script_path, new_emotion):
    try:
        # Read the script
        with open(script_path, 'r') as file:
            script_content = file.read()

        # Define explicit replacements
        script_content = script_content.replace(f'training/EI-oc-En-{old_emotion}-train.txt', train_filename)
        script_content = script_content.replace(f'development/2018-EI-oc-En-{old_emotion}-dev.txt', dev_filename)
        script_content = script_content.replace(f'test-gold/2018-EI-oc-En-{old_emotion}-test-gold.txt', test_filename)
        script_content = script_content.replace(f'generate_intensity_mapping("{old_emotion}")', f'generate_intensity_mapping("{new_emotion}")')
        script_content = script_content.replace( old_emotion, new_emotion)
        
        # Write modified script as a new file
        with open(new_script_path, 'w') as file:
            file.write(script_content)

        print(f"New script saved to {new_script_path}.")
    except Exception as e:
        print(f"An error occurred: {e}")

old_emotion = 'fear'
new_emotion = 'joy'
script_path = f'/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis/Notebook/{old_emotion}/ml_classification_{old_emotion}.py'
train_filename = f'training/EI-oc-En-{new_emotion}-train.txt'
dev_filename = f'development/2018-EI-oc-En-{new_emotion}-dev.txt'
test_filename = f'test-gold/2018-EI-oc-En-{new_emotion}-test-gold.txt'

new_script_path = f'/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis/Notebook/{new_emotion}/ml_classification_{new_emotion}.py'

replace_filenames_and_save_new(script_path, train_filename, dev_filename, test_filename, new_script_path, new_emotion)
