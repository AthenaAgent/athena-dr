#!/usr/bin/env python
"""
Simple test to verify GLM-4.5-air endpoint and MCP tools work together.
"""
import asyncio
import os
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from athena_dr.agent.workflows import AutoReasonSearchWorkflow

async def test_simple_query():
    """Test a simple query with the workflow."""
    
    config = {
        "tool_parser": "v20250824",
        "search_agent_model_name": "glm-4.5-air",
        "search_agent_api_key": "",  # No API key needed for your endpoint
        "search_agent_base_url": "https://trojanvectors--glm-inference-serve.modal.run/v1",
        "search_agent_max_tokens": 1000,
        "search_agent_temperature": 0.0,
        "search_agent_max_tool_calls": 2,
        "number_documents_to_search": 3,
        "search_timeout": 30,
        "browse_tool_name": "jina",
        "browse_timeout": 30,
        "browse_max_pages_to_fetch": 3,
        "browse_context_char_length": 2000,
        "use_browse_agent": False,
        "prompt_version": "v20250907",
        "mcp_transport_type": "FastMCPTransport",
    }
    
    print("Creating workflow...")
    workflow = AutoReasonSearchWorkflow(configuration=config)
    print("Workflow created successfully!")
    
    question = "What is the capital of France?"
    print(f"\nTesting with question: {question}")
    
    try:
        result = await workflow(
            problem=question,
            dataset_name="gaia",
            verbose=True
        )
        
        print("\n" + "="*80)
        print("SUCCESS!")
        print("="*80)
        print(f"Final answer: {result.get('final_response', 'No response')}")
        print(f"Tool calls made: {result.get('total_tool_calls', 0)}")
        print(f"Failed tool calls: {result.get('total_failed_tool_calls', 0)}")
        
        return result
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(test_simple_query())
    
    if result:
        print("\n✅ Test PASSED - Workflow is working!")
    else:
        print("\n❌ Test FAILED - Check errors above")
