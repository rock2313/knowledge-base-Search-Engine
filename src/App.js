import React, { useState } from 'react';
import { Search, Upload, FileText, Sparkles, Download, Trash2, AlertCircle, Loader2, BookOpen, Zap } from 'lucide-react';

export default function KnowledgeBaseSearch() {
    const [documents, setDocuments] = useState([]);
    const [query, setQuery] = useState('');
    const [answer, setAnswer] = useState(null);
    const [loading, setLoading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [searchMode, setSearchMode] = useState('search');
    const [selectedDocs, setSelectedDocs] = useState([]);

    const API_URL = 'http://localhost:8000';

    const handleFileUpload = async (e) => {
        const files = Array.from(e.target.files);

        for (const file of files) {
            if (file.type !== 'application/pdf' && !file.type.includes('text')) {
                alert(`${file.name} is not a supported file type`);
                continue;
            }

            const formData = new FormData();
            formData.append('file', file);

            try {
                setUploadProgress(10);
                const response = await fetch(`${API_URL}/upload`, {
                    method: 'POST',
                    body: formData,
                });

                setUploadProgress(50);
                const data = await response.json();

                if (response.ok) {
                    setDocuments(prev => [...prev, {
                        id: data.document_id,
                        name: file.name,
                        pages: data.pages,
                        chunks: data.chunks,
                        uploadedAt: new Date().toISOString()
                    }]);
                    setUploadProgress(100);
                } else {
                    alert(data.detail || 'Upload failed');
                }
            } catch (err) {
                alert('Upload failed: ' + err.message);
            } finally {
                setTimeout(() => setUploadProgress(0), 1000);
            }
        }
    };

    const handleSearch = async () => {
        if (!query.trim()) return;

        setLoading(true);
        setAnswer(null);

        try {
            const endpoint = searchMode === 'summarize' ? '/summarize' : '/search';
            const payload = {
                query: query,
                document_ids: selectedDocs.length > 0 ? selectedDocs : undefined,
                top_k: 5
            };

            const response = await fetch(`${API_URL}${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (response.ok) {
                setAnswer(data);
            } else {
                alert(data.detail || 'Search failed');
            }
        } catch (err) {
            alert('Search failed: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleDeleteDoc = async (docId) => {
        try {
            const response = await fetch(`${API_URL}/documents/${docId}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                setDocuments(prev => prev.filter(d => d.id !== docId));
                setSelectedDocs(prev => prev.filter(id => id !== docId));
            }
        } catch (err) {
            alert('Delete failed: ' + err.message);
        }
    };

    const toggleDocSelection = (docId) => {
        setSelectedDocs(prev =>
            prev.includes(docId)
                ? prev.filter(id => id !== docId)
                : [...prev, docId]
        );
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
            <div className="max-w-7xl mx-auto p-6">
                {/* Header */}
                <div className="mb-8 text-center">
                    <div className="flex items-center justify-center gap-3 mb-3">
                        <BookOpen className="w-12 h-12 text-purple-400" />
                        <h1 className="text-5xl font-bold text-white">Knowledge Base RAG</h1>
                    </div>
                    <p className="text-purple-200 text-lg">Advanced Document Search & Synthesis Engine</p>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Left Panel - Document Management */}
                    <div className="lg:col-span-1">
                        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                                <FileText className="w-6 h-6" />
                                Documents
                            </h2>

                            {/* Upload Button */}
                            <label className="block mb-6">
                                <div className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-semibold py-3 px-4 rounded-xl cursor-pointer transition-all flex items-center justify-center gap-2">
                                    <Upload className="w-5 h-5" />
                                    Upload Documents
                                </div>
                                <input
                                    type="file"
                                    multiple
                                    accept=".pdf,.txt"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                />
                            </label>

                            {uploadProgress > 0 && (
                                <div className="mb-4">
                                    <div className="bg-purple-900/50 rounded-full h-2 overflow-hidden">
                                        <div
                                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all"
                                            style={{ width: `${uploadProgress}%` }}
                                        />
                                    </div>
                                </div>
                            )}

                            {/* Document List */}
                            <div className="space-y-3 max-h-96 overflow-y-auto">
                                {documents.length === 0 ? (
                                    <div className="text-purple-300 text-center py-8">
                                        <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                                        No documents uploaded yet
                                    </div>
                                ) : (
                                    documents.map(doc => (
                                        <div
                                            key={doc.id}
                                            className={`bg-white/5 rounded-xl p-4 border transition-all cursor-pointer ${
                                                selectedDocs.includes(doc.id)
                                                    ? 'border-purple-400 bg-purple-500/20'
                                                    : 'border-white/10 hover:border-white/30'
                                            }`}
                                            onClick={() => toggleDocSelection(doc.id)}
                                        >
                                            <div className="flex items-start justify-between gap-2">
                                                <div className="flex-1 min-w-0">
                                                    <p className="text-white font-medium truncate">{doc.name}</p>
                                                    <p className="text-purple-300 text-sm mt-1">
                                                        {doc.pages} pages • {doc.chunks} chunks
                                                    </p>
                                                </div>
                                                <button
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        handleDeleteDoc(doc.id);
                                                    }}
                                                    className="text-red-400 hover:text-red-300 transition-colors"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                </button>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>

                            {selectedDocs.length > 0 && (
                                <div className="mt-4 p-3 bg-purple-500/20 rounded-lg border border-purple-400/30">
                                    <p className="text-purple-200 text-sm">
                                        <Zap className="w-4 h-4 inline mr-1" />
                                        {selectedDocs.length} document(s) selected
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right Panel - Search & Results */}
                    <div className="lg:col-span-2">
                        <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-6 border border-white/20">
                            <h2 className="text-2xl font-bold text-white mb-4 flex items-center gap-2">
                                <Sparkles className="w-6 h-6" />
                                Query Engine
                            </h2>

                            {/* Mode Selection */}
                            <div className="flex gap-2 mb-4">
                                <button
                                    onClick={() => setSearchMode('search')}
                                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                                        searchMode === 'search'
                                            ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                                            : 'bg-white/5 text-purple-300 hover:bg-white/10'
                                    }`}
                                >
                                    <Search className="w-4 h-4 inline mr-2" />
                                    Search
                                </button>
                                <button
                                    onClick={() => setSearchMode('summarize')}
                                    className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                                        searchMode === 'summarize'
                                            ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                                            : 'bg-white/5 text-purple-300 hover:bg-white/10'
                                    }`}
                                >
                                    <FileText className="w-4 h-4 inline mr-2" />
                                    Summarize
                                </button>
                            </div>

                            {/* Search Input */}
                            <div className="mb-6">
                                <div className="flex gap-2">
                                    <input
                                        type="text"
                                        value={query}
                                        onChange={(e) => setQuery(e.target.value)}
                                        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                                        placeholder={searchMode === 'search' ? "Ask anything about your documents..." : "Enter topic to summarize..."}
                                        className="flex-1 bg-white/5 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-purple-300 focus:outline-none focus:border-purple-400 focus:ring-2 focus:ring-purple-400/50"
                                        disabled={documents.length === 0}
                                    />
                                    <button
                                        onClick={handleSearch}
                                        disabled={loading || !query.trim() || documents.length === 0}
                                        className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold px-6 py-3 rounded-xl transition-all flex items-center gap-2 disabled:cursor-not-allowed"
                                    >
                                        {loading ? (
                                            <>
                                                <Loader2 className="w-5 h-5 animate-spin" />
                                                Processing
                                            </>
                                        ) : (
                                            <>
                                                <Search className="w-5 h-5" />
                                                {searchMode === 'search' ? 'Search' : 'Summarize'}
                                            </>
                                        )}
                                    </button>
                                </div>
                            </div>

                            {/* Results */}
                            {answer && (
                                <div className="space-y-4">
                                    {/* Main Answer */}
                                    <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-6 border border-purple-400/30">
                                        <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
                                            <Sparkles className="w-5 h-5 text-purple-400" />
                                            Synthesized Answer
                                        </h3>
                                        <p className="text-white leading-relaxed whitespace-pre-wrap">
                                            {answer.answer}
                                        </p>
                                    </div>

                                    {/* Source Chunks */}
                                    {answer.sources && answer.sources.length > 0 && (
                                        <div className="bg-white/5 rounded-xl p-6 border border-white/10">
                                            <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                                <FileText className="w-5 h-5 text-purple-400" />
                                                Source Excerpts ({answer.sources.length})
                                            </h3>
                                            <div className="space-y-3">
                                                {answer.sources.map((source, idx) => (
                                                    <div key={idx} className="bg-white/5 rounded-lg p-4 border border-white/10">
                                                        <div className="flex items-start justify-between gap-2 mb-2">
                              <span className="text-purple-400 font-medium text-sm">
                                {source.document_name} • Page {source.page}
                              </span>
                                                            <span className="text-purple-300 text-sm">
                                {Math.round(source.score * 100)}% match
                              </span>
                                                        </div>
                                                        <p className="text-purple-100 text-sm leading-relaxed">
                                                            {source.text}
                                                        </p>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {documents.length === 0 && (
                                <div className="bg-blue-500/10 border border-blue-400/30 rounded-xl p-6 text-center">
                                    <AlertCircle className="w-12 h-12 text-blue-400 mx-auto mb-3" />
                                    <p className="text-blue-200 font-medium mb-2">No Documents Available</p>
                                    <p className="text-blue-300 text-sm">Upload PDF or text documents to start searching</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}