'use client';

import React, { useState, useRef, useEffect } from 'react';

type Message = {
  role: 'user' | 'assistant';
  content: string;
};

export default function ChatInterface() {
  const [url, setUrl] = useState('');
  const [videoTitle, setVideoTitle] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [isProcessed, setIsProcessed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [status, setStatus] = useState('Ready');
  const [isDarkMode, setIsDarkMode] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [apiKey, setApiKey] = useState('');
  const [showSettings, setShowSettings] = useState(false);

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedSession = sessionStorage.getItem('chat_session_id');
      const savedTitle = sessionStorage.getItem('chat_video_title');

      if (savedSession) {
        setSessionId(savedSession);
        setIsProcessed(true);
        setStatus('Ready to chat (Active Session)');
        if (savedTitle) setVideoTitle(savedTitle);
      }
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedKey = sessionStorage.getItem('gemini_api_key');
      if (savedKey) setApiKey(savedKey);
    }
  }, []);

  const saveApiKey = (key: string) => {
    setApiKey(key);
    sessionStorage.setItem('gemini_api_key', key);
    setShowSettings(false);
  };

  const handleNewChat = () => {
    if (typeof window !== 'undefined') {
      sessionStorage.removeItem('chat_session_id');
      sessionStorage.removeItem('chat_video_title');
    }
    setSessionId('');
    setVideoTitle('');
    setUrl('');
    setMessages([]);
    setIsProcessed(false);
    setStatus('Ready');
    setInput('');
  };

  const handleProcess = async () => {
    if (!url) return;
    setIsLoading(true);
    setStatus('Processing video script...');

    try {
      const headers: any = { 'Content-Type': 'application/json' };
      if (apiKey) headers['x-gemini-api-key'] = apiKey;

      const res = await fetch(`${API_BASE}/process`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ url }),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Failed to process');

      setVideoTitle(data.video_title || 'Unknown Video');
      setIsProcessed(true);
      setSessionId(data.session_id);

      sessionStorage.setItem('chat_session_id', data.session_id);
      sessionStorage.setItem('chat_video_title', data.video_title || 'Unknown Video');

      setMessages([{ role: 'assistant', content: `Video processed! Ask me anything about "${data.video_title || 'this video'}".` }]);
      setStatus('Ready to chat');
    } catch (err: any) {
      alert(`Error: ${err.message}`);
      setStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setInput('');
    setIsLoading(true);
    setStatus('Thinking...');

    try {
      const headers: any = { 'Content-Type': 'application/json' };
      if (apiKey) headers['x-gemini-api-key'] = apiKey;

      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: headers,
        body: JSON.stringify({ question: userMsg, session_id: sessionId }),
      });

      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Failed to get answer');

      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }]);
      setStatus('Ready');
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}` }]);
      setStatus('Error');
    } finally {
      setIsLoading(false);
    }
  };

  const bgMain = isDarkMode ? 'bg-[#212121]' : 'bg-white';
  const bgSidebar = isDarkMode ? 'bg-[#171717]' : 'bg-[#F9F9F9]';
  const textPrimary = isDarkMode ? 'text-gray-100' : 'text-gray-800';
  const textSecondary = isDarkMode ? 'text-gray-400' : 'text-gray-500';
  const border = isDarkMode ? 'border-gray-700' : 'border-gray-200';
  const inputBg = isDarkMode ? 'bg-[#2f2f2f]' : 'bg-[#f4f4f4]';
  const userMsgBg = isDarkMode ? 'bg-[#2f2f2f]' : 'bg-[#f4f4f4]';

  return (
    <div className={`w-full h-full ${bgMain} ${textPrimary} flex font-sans text-sm md:text-base overflow-hidden`}>

      {showSettings && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className={`${bgMain} p-6 rounded-xl shadow-2xl w-full max-w-md border ${border}`}>
            <h3 className="text-lg font-semibold mb-2">Settings</h3>
            <p className={`text-sm ${textSecondary} mb-4`}>
              Enter your Google Gemini API Key. It is stored securely in your browser's session and never saved to our servers.
            </p>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Paste your API Key here"
              className={`w-full p-2 rounded-lg border ${border} ${inputBg} outline-none mb-4`}
            />
            <div className="flex justify-end space-x-2">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-gray-500 hover:text-gray-700 font-medium"
              >
                Cancel
              </button>
              <button
                onClick={() => saveApiKey(apiKey)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium"
              >
                Save Key
              </button>
            </div>
          </div>
        </div>
      )}

      <div className={`w-[260px] ${bgSidebar} flex flex-col p-3 transition-colors duration-300 hidden md:flex border-r ${border}`}>
        <div className="flex justify-between items-center mb-4 px-2">
          <button
            onClick={handleNewChat}
            className={`flex items-center space-x-2 text-sm font-medium ${textPrimary} hover:bg-gray-200/50 dark:hover:bg-gray-800 p-2 rounded-lg w-full transition-colors`}
          >
            <div className="p-1 border border-gray-300 dark:border-gray-600 rounded-full">
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>
            </div>
            <span>New chat</span>
          </button>

          <div className="flex items-center space-x-1">
            <button onClick={() => setShowSettings(true)} className="p-2 text-gray-500 hover:text-gray-800" title="API Key Settings">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto">
          <div className="px-3 py-2 text-xs font-semibold text-gray-400">Today</div>
          <div className={`cursor-pointer px-3 py-2 rounded-lg hover:bg-black/5 dark:hover:bg-white/5 truncate text-sm ${textPrimary}`}>
            {videoTitle || "New conversation"}
          </div>
        </div>

        <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 px-2 flex items-center space-x-3">
          <div className="w-8 h-8 bg-purple-600 rounded-full flex items-center justify-center text-white font-bold shadow-sm">
            U
          </div>
          <div className="font-medium text-sm">User</div>
        </div>
      </div>

      <div className="flex-1 flex flex-col relative h-full">

        {!isDarkMode && (
          <div className="absolute top-4 left-4 text-lg font-semibold text-gray-700 flex items-center space-x-1 z-10">
            <span>YouTube <span className="text-red-600">Chat AI</span></span>
          </div>
        )}

        <div className="flex-1 overflow-hidden relative flex flex-col items-center justify-center w-full">
          {!isProcessed ? (
            <div className="flex flex-col items-center space-y-8 mt-12 animate-in fade-in duration-500 p-4">
              <div className="w-16 h-16 bg-red-600 rounded-2xl flex items-center justify-center shadow-lg mb-2">
                <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" /></svg>
              </div>
              <h2 className={`text-2xl font-semibold ${textPrimary}`}>For the true clueless trying to make sense</h2>

              <div className="grid grid-cols-2 gap-4 w-full max-w-2xl">
                <button onClick={() => setUrl('#demo')} className={`text-left p-4 rounded-xl border ${border} hover:bg-gray-50 flex flex-col space-y-1 transition-all`}>
                  <span className={`font-medium ${textPrimary} text-sm`}>Summarize Video</span>
                  <span className={`text-xs ${textSecondary}`}>Get key takeaways instantly</span>
                </button>
                <button className={`text-left p-4 rounded-xl border ${border} hover:bg-gray-50 flex flex-col space-y-1 opacity-50 cursor-not-allowed`}>
                  <span className={`font-medium ${textPrimary} text-sm`}>Find Specific Topic</span>
                  <span className={`text-xs ${textSecondary}`}>Search within the video</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex-1 w-full max-w-3xl overflow-y-auto p-4 space-y-6 scroll-smooth pb-32">
              {messages.map((msg, idx) => (
                <div key={idx} className={`flex items-start space-x-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>

                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-full bg-red-600 flex-shrink-0 flex items-center justify-center text-white shadow-sm">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z" /></svg>
                    </div>
                  )}

                  <div className={`prose ${isDarkMode ? 'prose-invert' : ''} max-w-[85%] ${msg.role === 'user' ? `${userMsgBg} px-5 py-3 rounded-3xl` : ''}`}>
                    <p className={`whitespace-pre-wrap leading-relaxed ${textPrimary}`}>{msg.content}</p>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Area - Glassmorphism */}
        <div className="w-full max-w-3xl mx-auto px-4 pb-6 absolute bottom-0 left-0 right-0 z-10">
          <div className={`relative flex items-center ${isDarkMode ? 'bg-[#2f2f2f]/90' : 'bg-white/80'} backdrop-blur-md rounded-[26px] shadow-sm border ${border} px-4 py-3 transition-shadow focus-within:shadow-md`}>
            <svg className="w-6 h-6 text-gray-400 hover:text-gray-600 cursor-pointer mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4"></path></svg>

            <input
              type="text"
              value={!isProcessed ? url : input}
              onChange={(e) => !isProcessed ? setUrl(e.target.value) : setInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && (!isProcessed ? handleProcess() : handleSend())}
              placeholder={!isProcessed ? "Paste YouTube Video Link..." : "Ask me anything... (Check Settings to add API Key if getting errors)"}
              className={`flex-1 bg-transparent outline-none ${textPrimary} placeholder-gray-400`}
              disabled={isLoading}
            />

            <div className="flex items-center space-x-3 ml-2">
              {(!url && !input) && (
                <div className="p-2 rounded-full cursor-pointer hover:bg-black/5 dark:hover:bg-white/10">
                  <svg className="w-6 h-6 text-gray-800 dark:text-gray-200" fill="currentColor" viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" /><path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" /></svg>
                </div>
              )}

              {(url || input) && (
                <button
                  onClick={!isProcessed ? handleProcess : handleSend}
                  className={`p-1.5 rounded-lg bg-red-600 text-white hover:bg-red-700 ${isLoading ? 'opacity-50' : ''}`}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
