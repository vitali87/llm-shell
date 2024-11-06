import json

def process_file(content):
    # Split content into blocks separated by 'copy'
    blocks = content.split('\ncopy\n')
    
    # Initialize list to store command-description pairs
    command_pairs = []
    
    for block in blocks:
        if not block.strip():
            continue
            
        # Split each block into lines
        lines = block.strip().split('\n')
        
        if len(lines) < 2:
            continue
            
        # The first line is the instruction (description)
        instruction = lines[0].strip()
        
        # The second line is the command (output), remove leading '$ ' if present
        output = lines[1].strip()
        if output.startswith('$ '):
            output = output[2:]
            
        # Create a dictionary in the requested format
        command_dict = {
            "instruction": instruction,
            "output": output
        }
        command_pairs.append(command_dict)
    
    return command_pairs

def main():
    # Read input from file
    with open('paste.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Process the content
    command_pairs = process_file(content)
    
    # Write to JSONL file
    with open('commands.jsonl', 'w', encoding='utf-8') as outfile:
        for pair in command_pairs:
            json_line = json.dumps(pair, ensure_ascii=False)
            outfile.write(json_line + '\n')

if __name__ == "__main__":
    main()