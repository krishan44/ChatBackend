import os
import json
import logging
from datetime import datetime

class KnowledgeStore:
    def __init__(self):
        self.dataset_dir = os.path.join(os.getcwd(), 'Dataset')
        self.user_data_file = os.path.join(self.dataset_dir, 'user_contributed_data.json')
        self.ensure_directories()
        self.user_data = self.load_user_data()
        
    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            
    def load_user_data(self):
        """Load user-contributed data from file"""
        if os.path.exists(self.user_data_file):
            try:
                with open(self.user_data_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading user data: {str(e)}")
                return self.create_default_data()
        else:
            return self.create_default_data()
            
    def create_default_data(self):
        """Create default data structure"""
        return {
            "jobs": {},
            "queries": {},
            "metadata": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
    def save_user_data(self):
        """Save user data to file"""
        try:
            self.user_data["metadata"]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.user_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_data, f, indent=2)
            return True
        except Exception as e:
            logging.error(f"Error saving user data: {str(e)}")
            return False
            
    def add_new_job(self, job_title, data=None):
        """Add information about a new job"""
        if job_title not in self.user_data["jobs"]:
            self.user_data["jobs"][job_title] = {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "votes": 0,
                "data": data or {
                    "Degrees": [],
                    "Certificates": [],
                    "Skills": [],
                    "Roadmap": []
                }
            }
            return self.save_user_data()
        return False
            
    def update_job_data(self, job_title, category, item, user_id=None):
        """Add new data to an existing job"""
        if job_title not in self.user_data["jobs"]:
            self.add_new_job(job_title)
            
        if category not in self.user_data["jobs"][job_title]["data"]:
            self.user_data["jobs"][job_title]["data"][category] = []
            
        # Check if item already exists to avoid duplicates
        existing_items = set(self.user_data["jobs"][job_title]["data"][category])
        if item not in existing_items:
            self.user_data["jobs"][job_title]["data"][category].append({
                "item": item,
                "added_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "added_by": user_id or "anonymous",
                "votes": 1
            })
            return self.save_user_data()
        return False
            
    def vote_job_item(self, job_title, category, item, upvote=True):
        """Upvote or downvote a job item"""
        if job_title in self.user_data["jobs"] and category in self.user_data["jobs"][job_title]["data"]:
            for entry in self.user_data["jobs"][job_title]["data"][category]:
                if entry["item"] == item:
                    if upvote:
                        entry["votes"] += 1
                    else:
                        entry["votes"] -= 1
                    return self.save_user_data()
        return False
            
    def get_job_data(self, job_title, min_votes=2):
        """Get data about a job with items having at least min_votes"""
        result = {"Job": job_title}
        
        if job_title in self.user_data["jobs"]:
            for category, items in self.user_data["jobs"][job_title]["data"].items():
                # Filter by min_votes and extract just the item content
                reliable_items = [item["item"] for item in items if item["votes"] >= min_votes]
                if reliable_items:
                    result[category] = reliable_items
                    
        return result
            
    def record_query_response(self, query, response, helpful=True):
        """Record if a response was helpful for a query"""
        query_hash = str(hash(query))
        
        if query_hash not in self.user_data["queries"]:
            self.user_data["queries"][query_hash] = {
                "query": query,
                "responses": {}
            }
            
        response_hash = str(hash(response))
        if response_hash not in self.user_data["queries"][query_hash]["responses"]:
            self.user_data["queries"][query_hash]["responses"][response_hash] = {
                "response": response,
                "helpful_count": 0,
                "unhelpful_count": 0
            }
            
        if helpful:
            self.user_data["queries"][query_hash]["responses"][response_hash]["helpful_count"] += 1
        else:
            self.user_data["queries"][query_hash]["responses"][response_hash]["unhelpful_count"] += 1
            
        return self.save_user_data()

# Global instance
knowledge_store = KnowledgeStore()
