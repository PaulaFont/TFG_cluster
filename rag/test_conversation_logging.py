#!/usr/bin/env python3
"""
Test script to demonstrate conversation logging functionality.
This script shows how the conversation data is saved to CSV during RAG operations.
"""

import sys
import os
import pandas as pd

# Add the rag directory to Python path
sys.path.append('/home/pfont/rag')

def test_conversation_logging():
    """Test the conversation logging functionality"""
    
    try:
        from rag_core import RAGSystem
        
        print("=== Testing RAG Conversation Logging ===\n")
        
        # Initialize RAG system
        print("1. Initializing RAG system...")
        rag_system = RAGSystem()
        
        # Check if log file was created
        log_path = rag_system.get_conversation_log_path()
        print(f"2. Conversation log file: {log_path}")
        print(f"   Log file exists: {os.path.exists(log_path)}")
        
        if os.path.exists(log_path):
            # Read and display the CSV headers
            df = pd.read_csv(log_path)
            print(f"\n3. CSV Headers:")
            for i, col in enumerate(df.columns, 1):
                print(f"   {i:2d}. {col}")
            
            print(f"\n4. Current rows in log: {len(df)}")
            
            # Show sample conversation stats
            stats = rag_system.get_conversation_stats()
            print(f"\n5. Conversation Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        # Note: To test actual logging, you would need to:
        # 1. Initialize models and data: rag_system.initialize_models_and_data()
        # 2. Call chat_search with a test question
        print(f"\n6. Session ID: {rag_system.session_id}")
        print(f"   To see actual logging, run a full conversation through chat_search()")
        
        # Test JSON export functionality
        json_path = rag_system.export_conversation_log_as_json()
        if json_path:
            print(f"\n7. JSON export successful: {json_path}")
        
        print("\n=== Test completed successfully! ===")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required dependencies are installed.")
    except Exception as e:
        print(f"Error during testing: {e}")

def display_log_structure():
    """Display the expected structure of the conversation log"""
    
    print("\n=== Conversation Log Structure ===")
    print("The CSV file will contain the following columns:")
    
    columns = [
        ("timestamp", "ISO format timestamp of when the conversation occurred"),
        ("conversation_id", "Unique identifier for the conversation thread"),
        ("question", "The user's original question"),
        ("answer", "The system's response (cleaned of formatting)"),
        ("rag_answer_flag", "Boolean - whether RAG found relevant context"),
        ("graph_answer_flag", "Boolean - whether graph search found relevant info"),
        ("graph_answer_text", "The answer generated from graph context"),
        ("quien_entities", "WHO entities extracted from the query"),
        ("cuando_entities", "WHEN entities extracted from the query"),
        ("donde_entities", "WHERE entities extracted from the query"),
        ("que_entities", "WHAT entities extracted from the query"),
        ("search_terms", "Terms used for semantic search"),
        ("refined_query", "LLM-improved version of the user query"),
        ("document_id", "ID of the document that provided context"),
        ("document_filename", "Filename of the source document"),
        ("processing_version", "Version of document processing used"),
        ("cross_encoder_score", "Relevance score from cross-encoder"),
        ("graph_context_snippets", "Context snippets from graph search")
    ]
    
    for i, (col_name, description) in enumerate(columns, 1):
        print(f"{i:2d}. {col_name:25s} - {description}")

if __name__ == "__main__":
    display_log_structure()
    test_conversation_logging()
