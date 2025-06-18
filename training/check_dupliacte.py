import json
import re

def load_conversations(filepath):
    conversations_list = []
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            conversations_list.append(data["conversations"])
    return conversations_list

def normalize(text):
    # Remove all whitespace, newlines, and strip
    return re.sub(r'\s+', ' ', text).strip()

def message_to_tuple(msg):
    # Normalize the value for comparison
    return (msg["from"], normalize(msg["value"]))

def main():
    convs1 = load_conversations("combined_dataset.jsonl")
    convs2 = load_conversations("combined_dataset2.jsonl")

    # Flatten all messages in convs2 for fast lookup
    convs2_msgs = set()
    for conv in convs2:
        for msg in conv:
            convs2_msgs.add(message_to_tuple(msg))

    for idx, conv in enumerate(convs1):
        for jdx, msg in enumerate(conv):
            msg_tuple = message_to_tuple(msg)
            if msg_tuple in convs2_msgs:
                continue
                # print(f"Conversation {idx}, message {jdx} from combined_dataset.jsonl is present in combined_dataset2.jsonl")
            else:
                print(msg_tuple)
                print(f"Conversation {idx}, message {jdx} from combined_dataset.jsonl is NOT present in combined_dataset2.jsonl")
                break

if __name__ == "__main__":
    main()