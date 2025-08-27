import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.generative_models import grounding
import json

# Initialize Vertex AI
vertexai.init(project="your-gcp-project-id", location="us-central1")

class DeepSearchAgent:
    def __init__(self):
        # Initialize model with google search grounding
        self.google_search_tool = Tool.from_google_search_retrieval(
            grounding.GoogleSearchRetrieval()
        )
        
        self.model = GenerativeModel(
            "gemini-1.5-pro",
            tools=[self.google_search_tool]
        )
    
    def deep_search(self, query, search_depth=5):
        """
        Perform deep search - comprehensive results from multiple perspectives
        """
        
        # Stage 1: Basic search and analysis
        basic_prompt = f"""
        Search the web for the latest information about "{query}" and provide a comprehensive analysis.
        
        Please collect information from the following perspectives:
        1. Basic concepts and definitions
        2. Latest trends and developments
        3. Statistics and data
        4. Expert opinions and insights
        5. Real-world cases and current applications
        
        Critically evaluate all sources and provide a holistic and comprehensive overview.
        Clearly indicate the source for each piece of information.
        """
        
        print("🔍 Stage 1: Performing basic search...")
        basic_response = self.model.generate_content(basic_prompt)
        
        # Stage 2: In-depth analysis search
        deep_prompt = f"""
        Based on the previous search results for "{query}", perform additional searches on the following advanced topics:
        
        1. Comparative analysis of advantages and disadvantages of related technologies or methodologies
        2. Future prospects and predictions
        3. Related industry or market analysis
        4. Competitors or alternative solutions
        5. Regulatory or policy trends
        
        For each item, critically analyze the information and provide specific data with sources.
        Ensure the output is holistic and comprehensive, covering all relevant aspects.
        """
        
        print("🔍 Stage 2: Conducting in-depth analysis search...")
        deep_response = self.model.generate_content(deep_prompt)
        
        # Stage 3: Final comprehensive report generation
        final_prompt = f"""
        Synthesize all search results for "{query}" and create a high-quality report in the following format.
        Critically evaluate all information and ensure the final output is holistic and comprehensive:
        
        ## 🎯 Executive Summary
        
        ## 📊 Key Findings
        
        ## 🔄 Latest Trends and Developments
        
        ## 📈 Data and Statistics
        
        ## 💡 Expert Insights
        
        ## 🌟 Real-world Applications and Case Studies
        
        ## 🔮 Future Outlook
        
        ## ⚖️ Critical Analysis
        
        ## 📚 Key References
        
        For all information, cite reliable sources and prioritize currency and accuracy.
        Provide a critical, holistic and comprehensive analysis that considers multiple viewpoints and potential limitations.
        """
        
        print("🔍 Stage 3: Generating final comprehensive report...")
        final_response = self.model.generate_content(final_prompt)
        
        return {
            "basic_search": basic_response.text,
            "deep_analysis": deep_response.text,
            "final_report": final_response.text,
            "search_metadata": self._extract_sources(final_response)
        }
    
    def _extract_sources(self, response):
        """Extract grounding sources"""
        sources = []
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'grounding_metadata'):
                grounding_metadata = candidate.grounding_metadata
                if hasattr(grounding_metadata, 'grounding_chunks'):
                    for chunk in grounding_metadata.grounding_chunks:
                        if hasattr(chunk, 'web'):
                            sources.append({
                                "title": getattr(chunk.web, 'title', ''),
                                "uri": getattr(chunk.web, 'uri', ''),
                            })
        return sources

# Usage example
def run_deep_search(topic):
    """Execute deep search function"""
    try:
        agent = DeepSearchAgent()
        print(f"🚀 Starting deep search for '{topic}'...\n")
        
        results = agent.deep_search(topic)
        
        print("=" * 80)
        print("🎉 Deep search completed!")
        print("=" * 80)
        print("\n📋 Final Report:")
        print("-" * 50)
        print(results["final_report"])
        
        print("\n🔗 Key Reference Sources:")
        print("-" * 30)
        for i, source in enumerate(results["search_metadata"], 1):
            print(f"{i}. {source['title']}")
            print(f"   {source['uri']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

# Advanced search function with custom perspectives
def run_custom_deep_search(topic, perspectives=None):
    """
    Execute deep search with custom perspectives
    """
    if perspectives is None:
        perspectives = [
            "Technical implementation and architecture",
            "Market adoption and business impact", 
            "Regulatory and compliance considerations",
            "Competitive landscape analysis",
            "Future innovation potential"
        ]
    
    try:
        agent = DeepSearchAgent()
        print(f"🚀 Starting custom deep search for '{topic}' with {len(perspectives)} perspectives...\n")
        
        # Custom prompt with user-defined perspectives
        custom_prompt = f"""
        Conduct a comprehensive web search analysis for "{topic}" focusing on the following specific perspectives:
        
        {chr(10).join([f"{i+1}. {perspective}" for i, perspective in enumerate(perspectives)])}
        
        For each perspective, provide:
        - Current state and key developments
        - Critical analysis of strengths and limitations
        - Supporting data and evidence with sources
        - Expert opinions and industry insights
        
        Synthesize all findings into a holistic and comprehensive report that:
        - Critically evaluates conflicting information
        - Identifies gaps and areas needing further research  
        - Provides balanced, multi-faceted conclusions
        - Includes actionable insights and recommendations
        
        Ensure all sources are clearly cited and information is current and accurate.
        """
        
        response = agent.model.generate_content(
            custom_prompt,
            tools=[grounding.GoogleSearchRetrieval()]
        )
        
        result = {
            "custom_analysis": response.text,
            "search_metadata": agent._extract_sources(response),
            "perspectives_analyzed": perspectives
        }
        
        print("=" * 80)
        print("🎉 Custom deep search completed!")
        print("=" * 80)
        print(f"\n📋 Analysis covering {len(perspectives)} perspectives:")
        print("-" * 50)
        print(result["custom_analysis"])
        
        return result
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return None

# Execution example
if __name__ == "__main__":
    # User input
    research_topic = input("🔍 Enter the topic for deep search: ")
    
    # Choose search type
    search_type = input("Choose search type (1: Standard, 2: Custom): ")
    
    if search_type == "2":
        # Custom perspectives input
        print("Enter custom perspectives (press Enter after each, empty line to finish):")
        custom_perspectives = []
        while True:
            perspective = input("Perspective: ")
            if not perspective:
                break
            custom_perspectives.append(perspective)
        
        if custom_perspectives:
            results = run_custom_deep_search(research_topic, custom_perspectives)
        else:
            results = run_deep_search(research_topic)
    else:
        # Standard deep search
        results = run_deep_search(research_topic)
    
    # Save results (optional)
    if results:
        filename = f"deep_search_report_{research_topic.replace(' ', '_')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            if "final_report" in results:
                f.write(results["final_report"])
            else:
                f.write(results["custom_analysis"])
        print(f"\n💾 Report saved: {filename}")
