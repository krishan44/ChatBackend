import json
import os
import logging
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LearningSystem:
    def __init__(self, knowledge_file='learned_knowledge.json'):
        self.knowledge_file = os.path.join(os.getcwd(), 'Dataset', knowledge_file)
        self.learned_data = self._load_knowledge()
        self.feedback_threshold = 3  # Minimum feedback count to consider knowledge reliable
        self.confidence_threshold = 0.7  # Confidence threshold for using learned knowledge
        self.vectorizer = TfidfVectorizer()
        
    def _load_knowledge(self):
        """Load learned knowledge from file"""
        try:
            if os.path.exists(self.knowledge_file):
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create directories if they don't exist
                os.makedirs(os.path.dirname(self.knowledge_file), exist_ok=True)
                return {
                    "job_data": {},  # Job-related learned information
                    "queries": {},   # Common queries and responses
                    "feedback": {}   # Feedback on responses
                }
        except Exception as e:
            logging.error(f"Error loading knowledge file: {str(e)}")
            return {
                "job_data": {},
                "queries": {},
                "feedback": {}
            }
    
    def _save_knowledge(self):
        """Save learned knowledge to file"""
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.learned_data, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving knowledge file: {str(e)}")
    
    def store_user_correction(self, job, category, original_content, corrected_content):
        """Store user corrections to job information"""
        if job not in self.learned_data["job_data"]:
            self.learned_data["job_data"][job] = {}
        
        if category not in self.learned_data["job_data"][job]:
            self.learned_data["job_data"][job][category] = []
        
        # Check if this item already exists
        for item in self.learned_data["job_data"][job][category]:
            if item["content"] == corrected_content:
                item["confidence"] += 1
                self._save_knowledge()
                return
        
        # Add new correction
        correction = {
            "original": original_content,
            "content": corrected_content,
            "confidence": 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.learned_data["job_data"][job][category].append(correction)
        self._save_knowledge()
    
    def record_feedback(self, query, response, feedback_type):
        """Record user feedback for a response"""
        query_hash = str(hash(query))
        
        if query_hash not in self.learned_data["feedback"]:
            self.learned_data["feedback"][query_hash] = {
                "query": query,
                "positive": 0,
                "negative": 0,
                "responses": {}
            }
        
        response_hash = str(hash(response))
        if response_hash not in self.learned_data["feedback"][query_hash]["responses"]:
            self.learned_data["feedback"][query_hash]["responses"][response_hash] = {
                "response": response,
                "positive": 0,
                "negative": 0
            }
        
        # Update feedback count
        if feedback_type == "positive":
            self.learned_data["feedback"][query_hash]["positive"] += 1
            self.learned_data["feedback"][query_hash]["responses"][response_hash]["positive"] += 1
        else:
            self.learned_data["feedback"][query_hash]["negative"] += 1
            self.learned_data["feedback"][query_hash]["responses"][response_hash]["negative"] += 1
        
        self._save_knowledge()
    
    def learn_new_job_info(self, job, category, content):
        """Add new information about a job"""
        if job not in self.learned_data["job_data"]:
            self.learned_data["job_data"][job] = {}
        
        if category not in self.learned_data["job_data"][job]:
            self.learned_data["job_data"][job][category] = []
        
        # Check if similar content already exists using TF-IDF similarity
        if self.learned_data["job_data"][job][category]:
            existing_contents = [item["content"] for item in self.learned_data["job_data"][job][category]]
            
            # Only compute if we have content
            if existing_contents:
                try:
                    tfidf_matrix = self.vectorizer.fit_transform(existing_contents + [content])
                    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
                    
                    if np.max(cosine_sim) > 0.8:  # If very similar content exists
                        most_similar_idx = np.argmax(cosine_sim)
                        self.learned_data["job_data"][job][category][most_similar_idx]["confidence"] += 1
                        self._save_knowledge()
                        return
                except Exception as e:
                    logging.error(f"Error in similarity calculation: {str(e)}")
        
        # Add new content
        new_info = {
            "content": content,
            "confidence": 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        self.learned_data["job_data"][job][category].append(new_info)
        self._save_knowledge()
    
    def get_learned_job_data(self, job):
        """Get learned information about a job with sufficient confidence"""
        result = {}
        
        if job in self.learned_data["job_data"]:
            for category, items in self.learned_data["job_data"][job].items():
                # Filter items with confidence above threshold
                reliable_items = [item["content"] for item in items if item["confidence"] >= self.feedback_threshold]
                if reliable_items:
                    result[category] = reliable_items
        
        return result
    
    def learn_query_response(self, query, response, is_helpful=True):
        """Learn association between query and response"""
        query_hash = str(hash(query))
        
        if query_hash not in self.learned_data["queries"]:
            self.learned_data["queries"][query_hash] = {
                "query": query,
                "responses": {}
            }
        
        response_hash = str(hash(response))
        if response_hash not in self.learned_data["queries"][query_hash]["responses"]:
            self.learned_data["queries"][query_hash]["responses"][response_hash] = {
                "response": response,
                "helpful_count": 0,
                "unhelpful_count": 0
            }
        
        if is_helpful:
            self.learned_data["queries"][query_hash]["responses"][response_hash]["helpful_count"] += 1
        else:
            self.learned_data["queries"][query_hash]["responses"][response_hash]["unhelpful_count"] += 1
        
        self._save_knowledge()
    
    def find_similar_learned_query(self, query):
        """Find a similar query in the learned data using TF-IDF"""
        if not self.learned_data["queries"]:
            return None, None
        
        try:
            queries = [q_data["query"] for _, q_data in self.learned_data["queries"].items()]
            
            if not queries:
                return None, None
                
            tfidf_matrix = self.vectorizer.fit_transform(queries + [query])
            cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
            
            if np.max(cosine_sim) > self.confidence_threshold:
                most_similar_idx = np.argmax(cosine_sim)
                query_hash = str(hash(queries[most_similar_idx]))
                
                # Find best response based on helpful_count
                best_response = None
                max_helpful = -1
                
                for r_hash, r_data in self.learned_data["queries"][query_hash]["responses"].items():
                    helpful_ratio = r_data["helpful_count"] / max(1, r_data["helpful_count"] + r_data["unhelpful_count"])
                    if r_data["helpful_count"] > max_helpful and helpful_ratio > 0.6:
                        max_helpful = r_data["helpful_count"]
                        best_response = r_data["response"]
                
                return queries[most_similar_idx], best_response
            
            return None, None
        except Exception as e:
            logging.error(f"Error finding similar query: {str(e)}")
            return None, None

# Create a global instance of the learning system
learning_system = LearningSystem()
