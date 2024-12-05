# Add these imports at the top
import ollama
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import seaborn as sns

# Update the LLMInterface class
class LLMInterface:
    """Interface for interacting with Ollama LLM"""

    def __init__(self, model_name: str = "llama3.2:1b"):  # Corrected model name
        self.model_name = model_name

    async def generate_analysis(self, prompt: str, context: Optional[Dict] = None) -> str:
        try:
            response = ollama.generate(model=self.model_name, 
                                     prompt=prompt,
                                     context=context)
            return response['response']
        except Exception as e:
            logging.error(f"Error generating LLM response: {str(e)}")
            return "Error generating analysis"

class SeasonalityAgent(ReportAgent):
    """Agent for analyzing seasonal patterns in encounters"""

    def __init__(self, name: str):
        super().__init__(name)
        self.llm = LLMInterface()

    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Analyze monthly patterns
        monthly_trends = data.groupby('Month Grouping')['Encounter Count'].sum()

        # Create seasonal visualization
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=monthly_trends)
        plt.title('Seasonal Patterns in Border Encounters')
        plt.xlabel('Month')
        plt.ylabel('Total Encounters')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('seasonal_patterns.png')
        plt.close()

        # Calculate season-specific statistics
        seasons = {
            'Winter': ['December', 'January', 'February'],
            'Spring': ['March', 'April', 'May'],
            'Summer': ['June', 'July', 'August'],
            'Fall': ['September', 'October', 'November']
        }

        seasonal_stats = {}
        for season, months in seasons.items():
            seasonal_data = monthly_trends[monthly_trends.index.isin(months)]
            seasonal_stats[season] = {
                'total': seasonal_data.sum(),
                'average': seasonal_data.mean(),
                'peak_month': seasonal_data.idxmax()
            }

        # Generate LLM analysis
        analysis_prompt = f"""
        Analyze the following seasonal border encounter statistics and provide insights:
        {json.dumps(seasonal_stats, indent=2)}

        Focus on:
        1. Peak seasons and potential factors
        2. Seasonal patterns and trends
        3. Operational implications

        Provide a concise analysis in a professional tone.
        """

        llm_analysis = asyncio.run(self.llm.generate_analysis(analysis_prompt))

        return {
            'monthly_trends': monthly_trends.to_dict(),
            'seasonal_statistics': seasonal_stats,
            'visualization': 'seasonal_patterns.png',
            'llm_analysis': llm_analysis
        }

class EnforcementTypeAgent(ReportAgent):
    """Agent for analyzing enforcement types and components"""

    def __init__(self, name: str):
        super().__init__(name)
        self.llm = LLMInterface()

    def process_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        # Analyze enforcement components
        component_trends = data.groupby(['Fiscal Year', 'Component'])['Encounter Count'].sum()
        component_trends = component_trends.unstack(fill_value=0)

        # Create visualization
        plt.figure(figsize=(12, 6))
        component_trends.plot(kind='bar', stacked=True)
        plt.title('Enforcement Components Over Time')
        plt.xlabel('Fiscal Year')
        plt.ylabel('Encounter Count')
        plt.legend(title='Component', bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig('enforcement_trends.png')
        plt.close()

        # Calculate component-specific metrics
        component_metrics = {
            component: {
                'total_encounters': data[data['Component'] == component]['Encounter Count'].sum(),
                'yearly_growth': component_trends[component].pct_change().mean() * 100,
                'peak_year': component_trends[component].idxmax()
            }
            for component in data['Component'].unique()
        }

        # Generate LLM analysis
        analysis_prompt = f"""
        Analyze the following border enforcement statistics and provide insights:
        {json.dumps(component_metrics, indent=2)}

        Focus on:
        1. Effectiveness of different enforcement components
        2. Trends in enforcement methods
        3. Recommendations for resource allocation

        Provide a concise analysis in a professional tone.
        """

        llm_analysis = asyncio.run(self.llm.generate_analysis(analysis_prompt))

        return {
            'component_trends': component_trends.to_dict(),
            'component_metrics': component_metrics,
            'visualization': 'enforcement_trends.png',
            'llm_analysis': llm_analysis
        }

# Update the main function to include new agents
def main():
    # Create the reporting system
    system = ReportingSystem()

    # Add all agents including new ones
    system.add_agent(EncounterTrendsAgent("Trends Analyzer"))
    system.add_agent(CitizenshipAnalysisAgent("Citizenship Analyzer"))
    system.add_agent(DemographicAnalysisAgent("Demographics Analyzer"))
    system.add_agent(RegionalAnalysisAgent("Regional Analyzer"))
    system.add_agent(SeasonalityAgent("Seasonality Analyzer"))
    system.add_agent(EnforcementTypeAgent("Enforcement Analyzer"))

    # Load and prepare data
    try:
        data = pd.read_csv('nationwide-encounters-fy21-fy24-aor.csv')
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return

    system.load_data(data)

    # Generate reports
    reports = system.generate_reports()

    # Print reports
    for agent_name, report in reports.items():
        print(f"\n=== {agent_name} Report ===")
        print(report)

        # Save reports to files
        report_filename = f"reports/{agent_name.lower().replace(' ', '_')}_report.txt"
        with open(report_filename, 'w') as f:
            f.write(report)

if __name__ == "__main__":
    main()

# Created/Modified files during execution:
# - yearly_encounters.png
# - seasonal_patterns.png
# - enforcement_trends.png
# - reports/*.txt