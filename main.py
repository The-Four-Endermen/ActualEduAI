import os
import json
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Get API key securely
api_key = os.getenv('GEMINI_API_KEY')  # Change to match your .env file variable name

# Verify API key is available (without printing it)
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please add it to your .env file.")
else:
    logger.info("API key loaded successfully")

class MalaysianPrimaryAssessmentAnalyzer:
    def __init__(self, api_key):
        """Initialize the assessment analyzer with Gemini API key."""
        configure(api_key=api_key)
        self.model = GenerativeModel('gemini-1.5-pro')
        logger.info("Assessment analyzer initialized with Gemini 1.5 Pro model")
        
    def analyze_student_assessment(self, student_data):
        """
        Analyze student assessment data and provide recommendations.
        
        Args:
            student_data: Dictionary containing student assessment information
                {
                    "student_id": "S12345",
                    "grade_level": 4,
                    "subjects": {
                        "english": {
                            "overall_score": 75,
                            "components": {
                                "reading": 80,
                                "writing": 70,
                                "speaking": 75,
                                "listening": 75
                            }
                        },
                        "mathematics": {
                            "overall_score": 68,
                            "components": {
                                "arithmetic": 72,
                                "geometry": 60,
                                "problem_solving": 65,
                                "data_analysis": 75
                            }
                        }
                    }
                }
        
        Returns:
            Dictionary with analysis and recommendations
        """
        try:
            # Validate input data
            self._validate_student_data(student_data)
            
            # Prepare the prompt for Gemini
            prompt = self._create_analysis_prompt(student_data)
            
            # Call Gemini API
            logger.info(f"Sending request to Gemini API for student {student_data.get('student_id', 'unknown')}")
            response = self.model.generate_content(prompt)
            
            # Process and structure the response
            analysis = self._process_gemini_response(response.text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing student assessment: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}",
                "student_id": student_data.get("student_id", "unknown")
            }
    
    def _validate_student_data(self, data):
        """Validate the structure of student data."""
        required_fields = ["grade_level", "subjects"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
        required_subjects = ["english", "mathematics"]
        for subject in required_subjects:
            if subject not in data["subjects"]:
                raise ValueError(f"Missing required subject: {subject}")
                
        # Validate components exist for each subject
        for subject in required_subjects:
            if "components" not in data["subjects"][subject]:
                raise ValueError(f"Missing components for subject: {subject}")
    
    def _create_analysis_prompt(self, student_data):
        """Create structured prompt for Gemini API based on student data."""
        prompt = f"""
        You are an educational assessment expert specialized in Malaysian primary education curriculum.
        Analyze the following student assessment data and provide detailed recommendations:
        
        Student Grade Level: {student_data['grade_level']}
        
        English Assessment:
        - Overall Score: {student_data['subjects']['english']['overall_score']}
        - Reading: {student_data['subjects']['english']['components']['reading']}
        - Writing: {student_data['subjects']['english']['components']['writing']}
        - Speaking: {student_data['subjects']['english']['components']['speaking']}
        - Listening: {student_data['subjects']['english']['components']['listening']}
        
        Mathematics Assessment:
        - Overall Score: {student_data['subjects']['mathematics']['overall_score']}
        - Arithmetic: {student_data['subjects']['mathematics']['components']['arithmetic']}
        - Geometry: {student_data['subjects']['mathematics']['components']['geometry']}
        - Problem Solving: {student_data['subjects']['mathematics']['components']['problem_solving']}
        - Data Analysis: {student_data['subjects']['mathematics']['components']['data_analysis']}
        
        Please provide:
        1. Performance Level Classification (Low/Mid/High) for each subject with justification
        2. Key strengths identified across subjects
        3. Key weaknesses identified across subjects
        4. Specific curriculum recommendations and activities to improve weaknesses
        5. Suggested enrichment activities to further develop strengths
        
        Format your response as a structured JSON object with the following format:
        {{
            "performance_levels": {{
                "english": {{
                    "level": "High/Mid/Low",
                    "justification": "..."
                }},
                "mathematics": {{
                    "level": "High/Mid/Low",
                    "justification": "..."
                }}
            }},
            "strengths": [
                {{
                    "area": "...",
                    "description": "..."
                }}
            ],
            "weaknesses": [
                {{
                    "area": "...",
                    "description": "..."
                }}
            ],
            "improvement_recommendations": [
                {{
                    "target_area": "...",
                    "activities": [
                        "..."
                    ]
                }}
            ],
            "enrichment_activities": [
                {{
                    "target_strength": "...",
                    "activities": [
                        "..."
                    ]
                }}
            ]
        }}
        """
        return prompt
    
    def _process_gemini_response(self, response_text):
        """Process and structure the Gemini API response."""
        try:
            # First try to parse as JSON directly (if Gemini returned proper JSON)
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If not valid JSON, extract and parse JSON from text response
            # This handles cases where Gemini adds explanatory text around the JSON
            try:
                # Look for JSON-like content between curly braces
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON found, create structured output from the text
                    logger.warning("Could not find JSON structure in response")
                    return {
                        "error": "Could not parse JSON response",
                        "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
                    }
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                return {
                    "error": f"Error processing response: {str(e)}",
                    "raw_response": response_text[:500] + ("..." if len(response_text) > 500 else "")
                }

# Example usage
if __name__ == "__main__":
    try:
        # Initialize analyzer with API key from environment
        analyzer = MalaysianPrimaryAssessmentAnalyzer(api_key)
        
        # Sample student data
        sample_student = {
            "student_id": "S12345",
            "grade_level": 4,
            "subjects": {
                "english": {
                    "overall_score": 75,
                    "components": {
                        "reading": 80,
                        "writing": 70,
                        "speaking": 75,
                        "listening": 75
                    }
                },
                "mathematics": {
                    "overall_score": 68,
                    "components": {
                        "arithmetic": 72,
                        "geometry": 60,
                        "problem_solving": 65,
                        "data_analysis": 75
                    }
                }
            }
        }
        
        # Get recommendations
        logger.info(f"Analyzing assessment for student {sample_student['student_id']}")
        analysis_result = analyzer.analyze_student_assessment(sample_student)
        
        # Print the analysis result
        print("\n===== STUDENT ASSESSMENT ANALYSIS =====\n")
        print(json.dumps(analysis_result, indent=2))
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print(f"Error: {str(e)}")