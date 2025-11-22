import streamlit as st
import google.generativeai as genai
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List
import os
import re

# Data classes
@dataclass
class Task:
    description: str
    assignee: str
    due_date: str
    priority: str
    status: str = "Pending"

@dataclass
class Meeting:
    id: str
    title: str
    date: str
    duration: int
    attendees: list
    transcript: str = ""
    summary: str = ""
    tasks: list = field(default_factory=list)
    action_items: list = field(default_factory=list)
    decisions: list = field(default_factory=list)

# Initialize Gemini
def init_gemini(api_key: str) -> bool:
    """Initialize Google Gemini API."""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Gemini: {e}")
        return False

def get_gemini_model():
    """Get Gemini model instance."""
    # Use Gemini 2.0 Flash (stable) or 2.5 Flash for latest features
    return genai.GenerativeModel('gemini-2.0-flash')

# AI Processing Functions using Gemini
def extract_summary_ai(transcript: str, api_key: str) -> str:
    """Extract meeting summary using Gemini."""
    try:
        init_gemini(api_key)
        model = get_gemini_model()
        
        prompt = f"""Analyze this meeting transcript and provide a concise summary (2-3 sentences) covering the main topics discussed, key points, and overall meeting outcome.

Meeting Transcript:
{transcript}

Provide only the summary, no additional formatting or labels."""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def extract_tasks_ai(transcript: str, attendees: list, api_key: str) -> list:
    """Extract action items and tasks using Gemini."""
    try:
        init_gemini(api_key)
        model = get_gemini_model()
        
        attendees_str = ", ".join(attendees)
        today = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""Analyze this meeting transcript and extract all action items/tasks.

Meeting Transcript:
{transcript}

Attendees: {attendees_str}
Today's Date: {today}

Return a JSON array of tasks. Each task should have:
- "description": clear task description
- "assignee": person responsible (from attendees list, or "Unassigned")
- "due_date": in YYYY-MM-DD format (estimate based on context, default 7 days from today)
- "priority": "High", "Medium", or "Low" based on urgency

Return ONLY valid JSON array, no markdown, no explanation. Example:
[{{"description": "Complete report", "assignee": "Alice", "due_date": "2024-01-15", "priority": "High"}}]

If no tasks found, return: []"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        tasks_data = json.loads(response_text)
        tasks = []
        for t in tasks_data:
            tasks.append(Task(
                description=t.get("description", ""),
                assignee=t.get("assignee", "Unassigned"),
                due_date=t.get("due_date", (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")),
                priority=t.get("priority", "Medium")
            ))
        return tasks
    except json.JSONDecodeError:
        return []
    except Exception as e:
        st.warning(f"Task extraction error: {str(e)}")
        return []

def extract_decisions_ai(transcript: str, api_key: str) -> list:
    """Extract key decisions using Gemini."""
    try:
        init_gemini(api_key)
        model = get_gemini_model()
        
        prompt = f"""Analyze this meeting transcript and extract all key decisions that were made.

Meeting Transcript:
{transcript}

Return a JSON array of decision strings. Each decision should be a clear, concise statement.
Return ONLY valid JSON array, no markdown, no explanation. Example:
["Approved Q4 budget", "Decided to use new vendor"]

If no decisions found, return: []"""

        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean response
        response_text = re.sub(r'^```json\s*', '', response_text)
        response_text = re.sub(r'\s*```$', '', response_text)
        
        return json.loads(response_text)
    except:
        return []

def generate_followup_email_ai(meeting: Meeting, api_key: str) -> str:
    """Generate professional follow-up email using Gemini."""
    try:
        init_gemini(api_key)
        model = get_gemini_model()
        
        tasks_str = ""
        if meeting.tasks:
            for t in meeting.tasks:
                tasks_str += f"- {t.description} (Assignee: {t.assignee}, Due: {t.due_date}, Priority: {t.priority})\n"
        
        decisions_str = "\n".join([f"- {d}" for d in meeting.decisions]) if meeting.decisions else "None"
        
        prompt = f"""Generate a professional follow-up email for this meeting:

Meeting Title: {meeting.title}
Date: {meeting.date}
Attendees: {', '.join(meeting.attendees)}
Summary: {meeting.summary}
Decisions: {decisions_str}
Action Items: {tasks_str if tasks_str else 'None'}

Create a well-formatted, professional email that:
1. Thanks attendees
2. Summarizes key discussion points
3. Lists decisions made
4. Lists action items with assignees and due dates
5. Ends with a professional closing

Make it concise but comprehensive."""

        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating email: {str(e)}"

# Fallback functions (when API not available)
def extract_summary_fallback(transcript: str) -> str:
    """Fallback summary extraction without AI."""
    lines = transcript.strip().split('\n')
    topics = set()
    for line in lines:
        if ':' in line:
            content = line.split(':', 1)[1].strip().lower()
            if 'project' in content or 'update' in content:
                topics.add('Project Updates')
            if 'deadline' in content or 'timeline' in content:
                topics.add('Timeline Discussion')
            if 'budget' in content or 'cost' in content:
                topics.add('Budget Review')
            if 'issue' in content or 'problem' in content or 'blocker' in content:
                topics.add('Blockers & Issues')
    
    summary = f"Meeting covered {len(lines)} discussion points"
    if topics:
        summary += f" including: {', '.join(topics)}."
    return summary

def extract_tasks_fallback(transcript: str, attendees: list) -> list:
    """Fallback task extraction without AI."""
    tasks = []
    keywords = ['will', 'need to', 'should', 'action item', 'todo', 'task', 'follow up', 'deadline']
    lines = transcript.strip().split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in keywords):
            assignee = "Unassigned"
            for att in attendees:
                if att.lower() in line_lower or (': ' in line and att.lower() in line.split(':')[0].lower()):
                    assignee = att
                    break
            
            priority = "Medium"
            if 'urgent' in line_lower or 'asap' in line_lower or 'critical' in line_lower:
                priority = "High"
            elif 'when possible' in line_lower or 'eventually' in line_lower:
                priority = "Low"
            
            due = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            if 'tomorrow' in line_lower:
                due = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif 'end of week' in line_lower:
                due = (datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d")
            
            desc = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            tasks.append(Task(desc[:100], assignee, due, priority))
    
    return tasks

def extract_decisions_fallback(transcript: str) -> list:
    """Fallback decision extraction without AI."""
    decisions = []
    keywords = ['decided', 'agreed', 'approved', 'confirmed', 'will go with', 'final decision']
    
    for line in transcript.strip().split('\n'):
        if any(kw in line.lower() for kw in keywords):
            content = line.split(':', 1)[1].strip() if ':' in line else line.strip()
            decisions.append(content)
    
    return decisions

def generate_followup_email_fallback(meeting: Meeting) -> str:
    """Fallback email generation without AI."""
    email = f"""Subject: Meeting Summary - {meeting.title} ({meeting.date})

Hi Team,

Thank you for attending today's meeting. Here's a summary of what we discussed:

📋 SUMMARY
{meeting.summary}

"""
    if meeting.decisions:
        email += "✅ KEY DECISIONS\n"
        for i, dec in enumerate(meeting.decisions, 1):
            email += f"  {i}. {dec}\n"
        email += "\n"
    
    if meeting.tasks:
        email += "📌 ACTION ITEMS\n"
        for task in meeting.tasks:
            email += f"  • {task.description}\n"
            email += f"    Assignee: {task.assignee} | Due: {task.due_date} | Priority: {task.priority}\n"
        email += "\n"
    
    email += f"""👥 ATTENDEES
{', '.join(meeting.attendees)}

Please let me know if I missed anything or if you have any questions.

Best regards,
Meeting Agent 🤖
"""
    return email

# Initialize session state
if "meetings" not in st.session_state:
    st.session_state.meetings = []
if "current_meeting" not in st.session_state:
    st.session_state.current_meeting = None
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""

# Streamlit UI
st.set_page_config(page_title="Meeting Agent", page_icon="📅", layout="wide")

st.title("🤖 Meeting Agent")
st.caption("Auto-join meetings, take minutes, extract tasks, and send follow-ups")

# Sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    
    use_ai = st.checkbox("Use AI Processing", value=bool(st.session_state.api_key), disabled=not st.session_state.api_key)
    
    if st.session_state.api_key:
        st.success("✅ API Key configured")
    else:
        st.warning("⚠️ No API key - using basic extraction")
    
    st.divider()
    st.header("⚙️ Meeting Setup")
    
    meeting_type = st.selectbox("Meeting Type", ["Team Meeting", "Client Call", "Daily Stand-up", "1:1", "Custom"])
    meeting_title = st.text_input("Meeting Title", f"{meeting_type} - {datetime.now().strftime('%b %d')}")
    meeting_date = st.date_input("Date", datetime.now())
    meeting_time = st.time_input("Time", datetime.now())
    duration = st.slider("Duration (minutes)", 15, 120, 30)
    
    attendees_input = st.text_area("Attendees (one per line)", "Alice\nBob\nCharlie")
    attendees = [a.strip() for a in attendees_input.split('\n') if a.strip()]
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙️ Start" if not st.session_state.is_recording else "⏹️ Stop", use_container_width=True):
            st.session_state.is_recording = not st.session_state.is_recording
            if st.session_state.is_recording:
                st.session_state.current_meeting = Meeting(
                    id=datetime.now().strftime("%Y%m%d%H%M%S"),
                    title=meeting_title,
                    date=meeting_date.strftime("%Y-%m-%d"),
                    duration=duration,
                    attendees=attendees
                )
    with col2:
        if st.button("📋 New", use_container_width=True):
            st.session_state.current_meeting = None
            st.session_state.is_recording = False

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📝 Live Meeting", "📊 Analysis", "📧 Follow-up", "📚 History"])

with tab1:
    if st.session_state.is_recording:
        st.success("🔴 Recording in progress...")
    
    st.subheader("Meeting Transcript")
    
    transcript = st.text_area(
        "Enter or paste meeting transcript",
        height=300,
        placeholder="""Example format:
Alice: Let's discuss the project timeline.
Bob: I will complete the API integration by Friday.
Charlie: We decided to use the new design system.
Alice: Action item - Bob needs to update the documentation by end of week.
Bob: There's a blocker with the database migration, need urgent help.
Charlie: I agreed to handle the client presentation next Tuesday.""",
        key="transcript_input"
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        process_btn = st.button("🔍 Process Transcript", use_container_width=True, type="primary")
    with col2:
        save_btn = st.button("💾 Save Meeting", use_container_width=True)
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.session_state.current_meeting = None
        st.rerun()
    
    if process_btn and transcript:
        with st.spinner("Processing transcript with AI..." if use_ai and st.session_state.api_key else "Processing transcript..."):
            if st.session_state.current_meeting is None:
                st.session_state.current_meeting = Meeting(
                    id=datetime.now().strftime("%Y%m%d%H%M%S"),
                    title=meeting_title,
                    date=meeting_date.strftime("%Y-%m-%d"),
                    duration=duration,
                    attendees=attendees
                )
            
            meeting = st.session_state.current_meeting
            meeting.transcript = transcript
            
            if use_ai and st.session_state.api_key:
                meeting.summary = extract_summary_ai(transcript, st.session_state.api_key)
                meeting.tasks = extract_tasks_ai(transcript, attendees, st.session_state.api_key)
                meeting.decisions = extract_decisions_ai(transcript, st.session_state.api_key)
            else:
                meeting.summary = extract_summary_fallback(transcript)
                meeting.tasks = extract_tasks_fallback(transcript, attendees)
                meeting.decisions = extract_decisions_fallback(transcript)
            
            st.success("✅ Transcript processed successfully!")
    
    if save_btn and st.session_state.current_meeting:
        meeting = st.session_state.current_meeting
        meeting.transcript = transcript
        if not meeting.summary:
            if use_ai and st.session_state.api_key:
                meeting.summary = extract_summary_ai(transcript, st.session_state.api_key)
                meeting.tasks = extract_tasks_ai(transcript, attendees, st.session_state.api_key)
                meeting.decisions = extract_decisions_ai(transcript, st.session_state.api_key)
            else:
                meeting.summary = extract_summary_fallback(transcript)
                meeting.tasks = extract_tasks_fallback(transcript, attendees)
                meeting.decisions = extract_decisions_fallback(transcript)
        
        existing = [m for m in st.session_state.meetings if m.id != meeting.id]
        existing.append(meeting)
        st.session_state.meetings = existing
        st.success("✅ Meeting saved!")

with tab2:
    meeting = st.session_state.current_meeting
    
    if meeting and meeting.summary:
        st.subheader("📋 Meeting Summary")
        st.info(meeting.summary)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("✅ Key Decisions")
            if meeting.decisions:
                for dec in meeting.decisions:
                    st.markdown(f"• {dec}")
            else:
                st.caption("No decisions extracted")
        
        with col2:
            st.subheader("📊 Meeting Stats")
            st.metric("Attendees", len(meeting.attendees))
            st.metric("Tasks Extracted", len(meeting.tasks) if meeting.tasks else 0)
            st.metric("Decisions Made", len(meeting.decisions) if meeting.decisions else 0)
        
        st.divider()
        st.subheader("📌 Action Items & Tasks")
        
        if meeting.tasks:
            for i, task in enumerate(meeting.tasks):
                with st.expander(f"**{task.description[:50]}...**" if len(task.description) > 50 else f"**{task.description}**"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"👤 **Assignee:** {task.assignee}")
                    with col2:
                        st.markdown(f"📅 **Due:** {task.due_date}")
                    with col3:
                        priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                        st.markdown(f"**Priority:** {priority_colors.get(task.priority, '⚪')} {task.priority}")
        else:
            st.caption("No tasks extracted yet. Process a transcript first.")
    else:
        st.info("👆 Process a meeting transcript to see analysis")

with tab3:
    meeting = st.session_state.current_meeting
    
    if meeting and meeting.summary:
        st.subheader("📧 Follow-up Email")
        
        if use_ai and st.session_state.api_key:
            with st.spinner("Generating email with AI..."):
                email_content = generate_followup_email_ai(meeting, st.session_state.api_key)
        else:
            email_content = generate_followup_email_fallback(meeting)
        
        edited_email = st.text_area("Edit email before sending", email_content, height=400)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(edited_email)
                st.success("Email content displayed above - copy manually")
        with col2:
            if st.button("📤 Send Email", use_container_width=True, type="primary"):
                st.success("✅ Email sent successfully! (Simulated)")
        with col3:
            recipients = st.text_input("Recipients", ", ".join(meeting.attendees))
    else:
        st.info("👆 Process a meeting transcript to generate follow-up email")

with tab4:
    st.subheader("📚 Meeting History")
    
    if st.session_state.meetings:
        for meeting in reversed(st.session_state.meetings):
            with st.expander(f"**{meeting.title}** - {meeting.date}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"👥 **Attendees:** {len(meeting.attendees)}")
                with col2:
                    st.markdown(f"⏱️ **Duration:** {meeting.duration} min")
                with col3:
                    st.markdown(f"📌 **Tasks:** {len(meeting.tasks) if meeting.tasks else 0}")
                
                if meeting.summary:
                    st.markdown("**Summary:**")
                    st.caption(meeting.summary)
                
                if st.button(f"Load Meeting", key=f"load_{meeting.id}"):
                    st.session_state.current_meeting = meeting
                    st.rerun()
    else:
        st.caption("No meetings recorded yet")

# Footer
st.divider()
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("🤖 Meeting Agent v2.0 | Powered by Google Gemini AI | Made with Streamlit")
with col2:
    if st.session_state.api_key:
        st.caption("🟢 AI Enabled")
    else:
        st.caption("🟡 Basic Mode")
