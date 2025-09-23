"""
Comprehensive Testing Script for Neural Recommender System
==========================================================

This script tests the ModelInference class from inference.py with:
- Single prediction tests
- Batch prediction tests  
- Top-N recommendation tests
- Edge case handling tests

Author: Recommendation System Testing Suite
"""

import sys
import traceback
from typing import List, Tuple, Optional, Any
from inference import ModelInference

# Test Configuration
TEST_USERS = [1, 50, 999]
TEST_ITEMS = [10, 500, 9999]
TOP_N = 5
RUN_ID = "ef7c3a3244654413b979af93d39253e9"  # Latest run ID

class RecommenderTester:
    """Comprehensive test suite for the recommendation system"""
    
    def __init__(self, model_name: str = "neural_recommender", run_id: Optional[str] = None):
        """Initialize the tester with model configuration"""
        self.model_name = model_name
        self.run_id = run_id
        self.inference = None
        self.test_results = {
            "single_predictions": [],
            "batch_predictions": [],
            "recommendations": [],
            "edge_cases": []
        }
    
    def setup_model(self) -> bool:
        """Setup and initialize the model inference"""
        print("🔧 SETTING UP MODEL")
        print("=" * 60)
        
        try:
            # Initialize inference class
            self.inference = ModelInference(model_name=self.model_name, run_id=self.run_id)
            print(f"✅ ModelInference initialized")
            
            # Try loading from registry first
            print("📦 Attempting to load from model registry...")
            if self.inference.load_model_from_registry("latest"):
                print("✅ Model loaded from registry successfully!")
            else:
                print("⚠️  Registry loading failed, trying run-based loading...")
                if self.run_id and self.inference.load_model_from_run(self.run_id):
                    print(f"✅ Model loaded from run {self.run_id}")
                else:
                    print("❌ Failed to load model from both registry and run!")
                    return False
            
            # Prepare encoders
            print("🔄 Preparing data encoders...")
            if self.inference.prepare_encoders():
                print("✅ Encoders prepared successfully!")
                return True
            else:
                print("❌ Failed to prepare encoders!")
                return False
                
        except Exception as e:
            print(f"❌ Setup failed with error: {e}")
            traceback.print_exc()
            return False
    
    def test_single_predictions(self) -> None:
        """Test single user-item prediction functionality"""
        print("\n🎯 SINGLE PREDICTION TESTS")
        print("=" * 60)
        
        for user_id in TEST_USERS:
            print(f"\n--- Testing User {user_id} ---")
            
            for item_id in TEST_ITEMS:
                try:
                    prediction = self.inference.predict_rating(user_id, item_id)
                    
                    if prediction is not None:
                        result = f"✅ User {user_id} → Item {item_id}: {prediction:.3f}"
                        print(f"  {result}")
                        self.test_results["single_predictions"].append({
                            "user_id": user_id,
                            "item_id": item_id,
                            "prediction": prediction,
                            "status": "success"
                        })
                    else:
                        result = f"⚠️  User {user_id} → Item {item_id}: No prediction (not in training data)"
                        print(f"  {result}")
                        self.test_results["single_predictions"].append({
                            "user_id": user_id,
                            "item_id": item_id,
                            "prediction": None,
                            "status": "not_found"
                        })
                        
                except Exception as e:
                    result = f"❌ User {user_id} → Item {item_id}: Error - {str(e)}"
                    print(f"  {result}")
                    self.test_results["single_predictions"].append({
                        "user_id": user_id,
                        "item_id": item_id,
                        "prediction": None,
                        "status": "error",
                        "error": str(e)
                    })
    
    def test_batch_predictions(self) -> None:
        """Test batch prediction functionality"""
        print("\n📦 BATCH PREDICTION TESTS")
        print("=" * 60)
        
        # Create all combinations of test users and items
        user_item_pairs = [(user, item) for user in TEST_USERS for item in TEST_ITEMS]
        
        print(f"Testing {len(user_item_pairs)} user-item pairs in batch...")
        print(f"Pairs: {user_item_pairs}")
        
        try:
            batch_predictions = self.inference.predict_batch(user_item_pairs)
            
            print(f"\n📊 Batch Prediction Results:")
            for i, (pair, prediction) in enumerate(zip(user_item_pairs, batch_predictions)):
                user_id, item_id = pair
                
                if prediction is not None:
                    result = f"  {i+1:2d}. User {user_id:3d} → Item {item_id:4d}: {prediction:6.3f} ⭐"
                    status = "success"
                else:
                    result = f"  {i+1:2d}. User {user_id:3d} → Item {item_id:4d}: No prediction ❌"
                    status = "not_found"
                
                print(result)
                self.test_results["batch_predictions"].append({
                    "user_id": user_id,
                    "item_id": item_id,
                    "prediction": prediction,
                    "status": status
                })
                
        except Exception as e:
            print(f"❌ Batch prediction failed: {e}")
            traceback.print_exc()
            self.test_results["batch_predictions"].append({
                "error": str(e),
                "status": "error"
            })
    
    def test_recommendations(self) -> None:
        """Test top-N recommendation functionality"""
        print(f"\n🎬 TOP-{TOP_N} RECOMMENDATION TESTS")
        print("=" * 60)
        
        for user_id in TEST_USERS:
            print(f"\n--- Recommendations for User {user_id} ---")
            
            try:
                recommendations = self.inference.recommend_movies_for_user(user_id, top_k=TOP_N)
                
                if recommendations:
                    print(f"✅ Found {len(recommendations)} recommendations:")
                    for i, (item_id, rating) in enumerate(recommendations, 1):
                        print(f"  {i}. Item {item_id}: {rating:.3f} ⭐")
                    
                    self.test_results["recommendations"].append({
                        "user_id": user_id,
                        "recommendations": recommendations,
                        "count": len(recommendations),
                        "status": "success"
                    })
                else:
                    print("⚠️  No recommendations generated")
                    self.test_results["recommendations"].append({
                        "user_id": user_id,
                        "recommendations": [],
                        "count": 0,
                        "status": "empty"
                    })
                    
            except Exception as e:
                print(f"❌ Recommendation failed: {e}")
                self.test_results["recommendations"].append({
                    "user_id": user_id,
                    "error": str(e),
                    "status": "error"
                })
    
    def test_edge_cases(self) -> None:
        """Test edge cases and error handling"""
        print("\n🚨 EDGE CASE TESTS")
        print("=" * 60)
        
        edge_test_cases = [
            ("Non-existent user", 999999, 1),
            ("Non-existent item", 1, 999999),
            ("Both non-existent", 999999, 999999),
            ("Negative user ID", -1, 1),
            ("Negative item ID", 1, -1),
            ("Zero user ID", 0, 1),
            ("Zero item ID", 1, 0)
        ]
        
        for test_name, user_id, item_id in edge_test_cases:
            print(f"\n--- {test_name}: User {user_id}, Item {item_id} ---")
            
            try:
                # Test single prediction
                prediction = self.inference.predict_rating(user_id, item_id)
                if prediction is not None:
                    print(f"  Single prediction: {prediction:.3f}")
                    status = "unexpected_success"
                else:
                    print(f"  Single prediction: No result (expected)")
                    status = "expected_failure"
                
                # Test recommendations for user edge cases
                if user_id >= 0:  # Only test recommendations for non-negative user IDs
                    recommendations = self.inference.recommend_movies_for_user(user_id, top_k=3)
                    print(f"  Recommendations: {len(recommendations) if recommendations else 0} items")
                else:
                    recommendations = []
                    print(f"  Recommendations: Skipped (negative user ID)")
                
                self.test_results["edge_cases"].append({
                    "test_name": test_name,
                    "user_id": user_id,
                    "item_id": item_id,
                    "prediction": prediction,
                    "recommendations_count": len(recommendations) if recommendations else 0,
                    "status": status
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                self.test_results["edge_cases"].append({
                    "test_name": test_name,
                    "user_id": user_id,
                    "item_id": item_id,
                    "error": str(e),
                    "status": "error"
                })
    
    def print_test_summary(self) -> None:
        """Print comprehensive test summary"""
        print("\n📊 TEST SUMMARY")
        print("=" * 60)
        
        # Single predictions summary
        single_success = sum(1 for r in self.test_results["single_predictions"] if r["status"] == "success")
        single_total = len(self.test_results["single_predictions"])
        print(f"Single Predictions: {single_success}/{single_total} successful")
        
        # Batch predictions summary
        batch_success = sum(1 for r in self.test_results["batch_predictions"] if r.get("status") == "success")
        batch_total = len([r for r in self.test_results["batch_predictions"] if "user_id" in r])
        print(f"Batch Predictions:  {batch_success}/{batch_total} successful")
        
        # Recommendations summary
        rec_success = sum(1 for r in self.test_results["recommendations"] if r["status"] == "success")
        rec_total = len(self.test_results["recommendations"])
        print(f"Recommendations:    {rec_success}/{rec_total} successful")
        
        # Edge cases summary
        edge_handled = sum(1 for r in self.test_results["edge_cases"] if r["status"] in ["expected_failure", "error"])
        edge_total = len(self.test_results["edge_cases"])
        print(f"Edge Cases:         {edge_handled}/{edge_total} handled properly")
        
        # Overall success rate
        total_tests = single_total + batch_total + rec_total + edge_total
        total_success = single_success + batch_success + rec_success + edge_handled
        success_rate = (total_success / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n🎯 Overall Success Rate: {success_rate:.1f}% ({total_success}/{total_tests})")
        
        # Performance insights
        successful_predictions = [r for r in self.test_results["single_predictions"] if r["status"] == "success"]
        if successful_predictions:
            ratings = [r["prediction"] for r in successful_predictions]
            avg_rating = sum(ratings) / len(ratings)
            min_rating = min(ratings)
            max_rating = max(ratings)
            print(f"\n📈 Prediction Statistics:")
            print(f"   Average Rating: {avg_rating:.3f}")
            print(f"   Rating Range:   {min_rating:.3f} - {max_rating:.3f}")
    
    def run_all_tests(self) -> bool:
        """Run the complete test suite"""
        print("🧪 NEURAL RECOMMENDER SYSTEM TEST SUITE")
        print("=" * 60)
        print(f"Test Users: {TEST_USERS}")
        print(f"Test Items: {TEST_ITEMS}")
        print(f"Top-N:      {TOP_N}")
        print(f"Run ID:     {self.run_id}")
        
        # Setup model
        if not self.setup_model():
            print("\n❌ Test suite aborted due to setup failure!")
            return False
        
        # Run all test categories
        try:
            self.test_single_predictions()
            self.test_batch_predictions()
            self.test_recommendations()
            self.test_edge_cases()
            self.print_test_summary()
            
            print("\n✅ Test suite completed successfully!")
            return True
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            traceback.print_exc()
            return False

def main():
    """Main function to run the test suite"""
    
    # Initialize and run tester
    tester = RecommenderTester(
        model_name="neural_recommender",
        run_id=RUN_ID
    )
    
    success = tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests completed! Your recommendation system is working properly.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()