import json

# moving the dataset files to a new file and renaming the keys
filenames = ["1.jsonl", "2.jsonl", "3.jsonl", "5.jsonl", "6.jsonl"] #the ones with proper conversation format
# filenames = ["combined_dataset2.jsonl"]
with open("combined_dataset.jsonl", "w") as file:
    
    for filename in filenames:
        with open("dataset/" + filename, "r") as f:
            for line in f:
                data = json.loads(line) # laod each line
                data["conversations"].pop(0) # remove the instruction
                # for i, conversation in enumerate(data["conversations"]):
                #     if conversation["from"] == "human": #this is not needed, but i just used it to see if the changes were happening
                #         conversation["from"] = "user"
                #     elif conversation["from"] == "gpt":
                #         conversation["from"] = "assistant"
                file.write(json.dumps(data) + "\n") # write the data to the new file
print("Combined dataset created successfully.")


filenames = ["4.jsonl", "7.jsonl", "8.jsonl", "9.jsonl", "10.jsonl"] #the ones with proper conversation format


for filename in filenames:
    with open("dataset/"+filename, "r") as f:
        for line in f:
            data = json.loads(line)
            # print(type(data))   
            conversations = data["text"] # here conversations is a string
            
            conversations_lst = [] #keep track of conversations
            
            while conversations:
                human_index = conversations.find("Human: ")
                finished_index = conversations.find("**Finished.**\n", human_index)
                print("Human index:", human_index)
                print("Finished index:", finished_index)
                if human_index == -1 or finished_index == -1:
                    print("Error: Could not find expected conversation markers.")
                    break
                # Extract the conversation part for human
                conversation_part = conversations[human_index + len("Human: "):finished_index].strip()
                human_dic = {"from": "human", "value": conversation_part}
                conversations_lst.append(human_dic)
                
                # Remove the processed part from conversations
                conversations = conversations[finished_index + len("**Finished.**"):].strip()
                # print("Remaining conversations:", conversations)
                
                ai_index = conversations.find("AI: ")
                finished_index = conversations.find("**Finished.**", ai_index)
                print("AI index:", ai_index)
                print("Finished index:", finished_index)
                if ai_index == -1 or finished_index == -1:
                    print("Error: Could not find expected conversation markers for AI.")
                    break
                # Extract the conversation part for AI
                conversation_part = conversations[ai_index + len("AI: "):finished_index].strip()
                # print("AI conversation part:", conversation_part)
                ai_dic = {"from": "ai", "value": conversation_part}
                # Now find the ai response
                conversations_lst.append(ai_dic)
                
                # Remove the processed part from conversations
                conversations = conversations[finished_index + len("**Finished.**"):].strip()
                
            # Now we have a list of conversations
            conv_dic = {"conversations": conversations_lst} 
            with open("combined_dataset2.jsonl", "a") as file:
                file.write(json.dumps(conv_dic) + "\n")
                
    
                