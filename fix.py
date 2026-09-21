import os
for file in os.listdir('.'):
    if file.endswith('.py') and file != 'bootstrap.py':
        with open(file, 'r') as f:
            text = f.read()
        if text.endswith('\\n'):
            with open(file, 'w') as f:
                f.write(text[:-2] + '\n')
print("Files fixed!")

