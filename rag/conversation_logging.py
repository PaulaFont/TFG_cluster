def get_conversation_log_path(self):
        """Return the path to the conversation log file"""
        return self.conversations_log_path

def export_conversation_log_as_json(self, output_path=None):
    """Export conversation log as JSON format"""
    import csv
    import json
    
    if not output_path:
        json_filename = f"conversations_log_{self.session_id}.json"
        output_path = os.path.join(self.BASE_DOCUMENT_DIRECTORY, json_filename)
    
    try:
        conversations = []
        if os.path.exists(self.conversations_log_path):
            with open(self.conversations_log_path, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    conversations.append(dict(row))
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(conversations, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"Exported conversation log to JSON: {output_path}")
            return output_path
        else:
            print(f"No conversation log file found at: {self.conversations_log_path}")
            return None
            
    except Exception as e:
        print(f"Error exporting conversation log to JSON: {e}")
        return None

def get_conversation_stats(self):
    """Get basic statistics about the conversations in this session"""
    import csv
    
    try:
        if not os.path.exists(self.conversations_log_path):
            return {"total_conversations": 0, "rag_answers": 0, "graph_answers": 0}
        
        stats = {
            "total_conversations": 0,
            "rag_answers": 0, 
            "graph_answers": 0,
            "conversations_with_both": 0,
            "unique_documents": set(),
            "most_common_entities": {}
        }
        
        with open(self.conversations_log_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                stats["total_conversations"] += 1
                
                if row["rag_answer_flag"].lower() == "true":
                    stats["rag_answers"] += 1
                
                if row["graph_answer_flag"].lower() == "true":  
                    stats["graph_answers"] += 1
                    
                if (row["rag_answer_flag"].lower() == "true" and 
                    row["graph_answer_flag"].lower() == "true"):
                    stats["conversations_with_both"] += 1
                
                if row["document_id"]:
                    stats["unique_documents"].add(row["document_id"])
        
        stats["unique_documents"] = len(stats["unique_documents"])
        return stats
        
    except Exception as e:
        print(f"Error getting conversation stats: {e}")
        return {"error": str(e)}