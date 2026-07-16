from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
# Import our clean merged classes from agent.py
from agent import AgentRouter, FAQAgent, BillingAgent

app = FastAPI(title="Multi-Agent AI Hub Server")

# Instantiate components smoothly
router = AgentRouter()
faq_agent = FAQAgent()
billing_agent = BillingAgent()

# Modern Gemini SDK initialization
GEMINI_API_KEY = "AQ.Ab8RN6LFdv45V162pzEi3yDrxG-99IGcj98ajJAzyeW_9pj_SA"
client = genai.Client(api_key=GEMINI_API_KEY)

class QueryRequest(BaseModel):
    text: str

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    query = request.text
    
    # 1. Routing Decision Execution
    intent, confidence = router.route_query(query)
    
    context = ""
    agent_log = ""
    
    # 2. Specialized Agent Tracing Logic
    if intent == "faq":
        context = faq_agent.retrieve_context(query)
        agent_log = "FAQ Agent pulled relevant contexts from ChromaDB vector rows."
    elif intent == "billing":
        context = billing_agent.process_billing_action(query)
        agent_log = "Billing Agent scanned local CSV templates."
    elif intent == "technical":
        context = "Diagnostics: Check system access status tokens, flush active state caches."
        agent_log = "Technical support automation rules deployed."
    else:
        context = "Default company guidelines applied."
        agent_log = "General inquiry fallback activated."

    # 3. Smart LLM Synthesizer Layer
    try:
        prompt = f"""
        You are a capstone grade Multi-Agent master customer support network. Answer the User Query expertly.
        Synthesize the system context if useful, but leverage your knowledge base to respond cleanly if context is broad.

        System Context: {context}
        User Query: {query}
        """
        
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        ai_response = response.text
    except Exception as e:
        ai_response = f"Synthesis workflow halted. Raw details: {str(e)}"

    return {
        "intent": intent,
        "confidence": confidence,
        "log": agent_log,
        "context": context,
        "response": ai_response
    }