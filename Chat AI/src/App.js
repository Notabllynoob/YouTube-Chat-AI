
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';


const SESSION_ID = `session_${Date.now()}`;
const API_BASE_URL = 'http://127.0.0.1:8000';


const YouTubeIcon = () => (
  <svg height="24px" width="24px" version="1.1" id="Layer_1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 225 225">
    <g>
      <path style={{ fill: '#FF0000' }} d="M213.2,59.3c-2.4-8.8-9-15.5-17.8-17.9C180.7,37.5,112.5,37.5,112.5,37.5s-68.2,0-82.9,3.9 C20.8,43.8,14.1,50.5,11.8,59.3C7.5,74.1,7.5,112.5,7.5,112.5s0,38.4,4.3,53.2c2.4,8.8,9,15.5,17.8,17.9 C44.3,187.5,112.5,187.5,112.5,187.5s68.2,0,82.9-3.9c8.8-2.4,15.5-9,17.8-17.9c4.3-14.8,4.3-53.2,4.3-53.2 S217.5,74.1,213.2,59.3z" />
      <polygon style={{ fill: '#FFFFFF' }} points="90,143.2 148.8,112.5 90,81.8" />
    </g>
  </svg>
);

const SpinnerIcon = () => (
  <svg className="spinner" viewBox="0 0 50 50">
    <circle className="path" cx="25" cy="25" r="20" fill="none" strokeWidth="5"></circle>
  </svg>
);


function App() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [question, setQuestion] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [processingStatus, setProcessingStatus] = useState('idle');

  const handleProcessVideo = async () => {
    if (!youtubeUrl) {
      alert('Please enter a YouTube URL.');
      return;
    }
    setProcessingStatus('loading');
    setChatHistory([]);
    try {
      await axios.post(`${API_BASE_URL}/process-video`, {
        youtube_url: youtubeUrl,
        session_id: SESSION_ID,
      });
      setProcessingStatus('success');
    } catch (error) {
      setProcessingStatus('error');
      console.error(error);
    }
  };

  const handleAskQuestion = async () => {
    if (!question) return;

    const newChatHistory = [...chatHistory, { type: 'user', message: question }];
    setChatHistory(newChatHistory);
    setQuestion('');

    try {
      const response = await axios.post(`${API_BASE_URL}/ask-question`, {
        question: question,
        session_id: SESSION_ID,
      });
      setChatHistory([...newChatHistory, { type: 'bot', message: response.data.answer }]);
    } catch (error) {
      setChatHistory([...newChatHistory, { type: 'bot', message: 'Sorry, an error occurred.' }]);
      console.error(error);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>YouTube AI Assistant</h1>
        <p className="subtitle">Paste a YouTube link to start asking questions about the video.</p>
      </header>

      <div className="url-processor">
        <div className="input-wrapper">
          <YouTubeIcon />
          <input
            type="text"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
            disabled={processingStatus === 'loading'}
          />
        </div>
        <button onClick={handleProcessVideo} disabled={processingStatus === 'loading'}>
          {processingStatus === 'loading' ? <SpinnerIcon /> : 'Process Video'}
        </button>
      </div>

      {processingStatus === 'success' && <div className="feedback-message success">✅ Video processed! You can now ask questions below.</div>}
      {processingStatus === 'error' && <div className="feedback-message error">❌ Failed to process video. Please check the URL and try again.</div>}

      <div className="chat-container">
        {chatHistory.map((entry, index) => (
          <div key={index} className={`chat-message ${entry.type}`}>
            <p>{entry.message}</p>
          </div>
        ))}
      </div>

      <div className="question-input">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={processingStatus === 'success' ? "Ask a question..." : "Process a video first"}
          disabled={processingStatus !== 'success'}
          onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
        />
        <button onClick={handleAskQuestion} disabled={processingStatus !== 'success'}>
          Send
        </button>
      </div>
    </div>
  );
}

export default App;