'use client';

import React, { useState, useEffect } from 'react';
import DotMatrix from '../components/DotMatrix';
import ImageUploader from '../components/ImageUploader';
import ProcessingLoader from '../components/ProcessingLoader';

const BACKEND_URL = 'http://localhost:8000';

export default function DehazingDashboard() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<{ original: string; dehazed: string; haze_level: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const processImage = async () => {
    if (!selectedImage) return;
    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedImage);

    try {
      const response = await fetch(`${BACKEND_URL}/dehaze`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      setResult({
        original: previewUrl!,
        dehazed: `data:image/png;base64,${data.image_base64}`,
        haze_level: data.haze_level,
      });
    } catch (err: any) {
      setError('Connection failed. Verify backend at port 8000.');
      console.error(err);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main className="min-h-screen relative flex flex-col p-8 md:p-12 items-center bg-dots selection:bg-blue-100">
      <DotMatrix />

      {/* Header */}
      <header className="mb-14 text-center">
        <h1 className="text-[32px] font-normal tracking-tight text-[#1e293b] mb-3">Adaptive Image Dehazing</h1>
        <p className="text-sm text-slate-500 font-light tracking-wide opacity-80">
          Weather-conditioned neural dehazing | Research Prototype
        </p>
      </header>

      {/* Main Grid */}
      <div className="w-full max-w-[1240px] grid grid-cols-1 lg:grid-cols-12 gap-8 items-stretch">

        {/* PANEL 1: SYSTEM OVERVIEW */}
        <section className="lg:col-span-4 panel-container p-6 bg-[#f8fafc]">
          <h2 className="text-sm font-bold text-slate-700 mb-8 tracking-tight">System Overview</h2>

          <div className="flex-1 flex flex-col items-center justify-center relative min-h-[500px]">
            <svg viewBox="0 0 160 380" className="w-full h-full max-w-[300px]">
              <defs>
                <marker id="arrow" markerWidth="8" markerHeight="5" refX="7" refY="2.5" orient="auto">
                  <polygon points="0 0, 8 2.5, 0 5" fill="#94a3b8" />
                </marker>
              </defs>

              {/* Web App */}
              <rect x="5" y="10" width="150" height="40" rx="2" fill="white" stroke="#e2e8f0" strokeWidth="1" />
              <text x="80" y="32" textAnchor="middle" fontSize="6.5" fill="#475569" className="font-medium tracking-wide">Web Application (Next.js)</text>

              <path d="M80 50 V75" stroke="#94a3b8" strokeWidth="0.8" markerEnd="url(#arrow)" />
              <text x="85" y="65" fontSize="4.5" fill="#94a3b8" className="italic opacity-70">POST</text>

              {/* Backend Engine */}
              <rect x="5" y="80" width="150" height="50" rx="2" fill="white" stroke="#e2e8f0" strokeWidth="1" />
              <text x="80" y="105" textAnchor="middle" fontSize="6.5" fill="#475569" className="font-medium">FastAPI Inference Engine</text>
              <text x="80" y="117" textAnchor="middle" fontSize="6" fill="#64748b" className="opacity-80">(Python + CUDA)</text>

              <path d="M80 130 V155" stroke="#94a3b8" strokeWidth="0.8" markerEnd="url(#arrow)" />

              {/* PyTorch Model Panel */}
              <rect x="5" y="160" width="150" height="180" rx="10" fill="white" stroke="#e2e8f0" strokeWidth="1" />
              <circle cx="20" cy="175" r="5" fill="none" stroke="#94a3b8" strokeWidth="1" />
              <text x="20" y="177" textAnchor="middle" fontSize="4.5" fill="#475569">ⓘ</text>
              <text x="32" y="177" fontSize="7" fill="#475569" className="font-bold">PyTorch Model</text>

              {/* ResNet */}
              <rect x="15" y="195" width="130" height="50" rx="2" fill="white" stroke="#cbd5e1" strokeWidth="0.8" />
              <image href="https://img.icons8.com/material-outlined/24/475569/image.png" x="25" y="205" width="10" height="10" />
              <text x="82" y="213" textAnchor="middle" fontSize="6" fill="#475569" className="font-medium">ResNet-18 Feature Extractor</text>
              <text x="82" y="226" textAnchor="middle" fontSize="5" fill="#94a3b8" className="opacity-80">(Frozen Weights)</text>

              <path d="M80 245 V265" stroke="#94a3b8" strokeWidth="0.8" markerEnd="url(#arrow)" />

              {/* U-Net */}
              <rect x="15" y="270" width="130" height="60" rx="2" fill="white" stroke="#cbd5e1" strokeWidth="0.8" />
              <image href="https://img.icons8.com/material-outlined/24/475569/grid.png" x="25" y="280" width="10" height="10" />
              <text x="82" y="288" textAnchor="middle" fontSize="6" fill="#475569" className="font-bold">Conditional U-Net</text>
              <text x="82" y="300" textAnchor="middle" fontSize="5" fill="#94a3b8">Haze Severity Clustering</text>
              <path d="M35 315 L42 315" stroke="#94a3b8" strokeWidth="0.5" markerEnd="url(#arrow)" />
              <text x="50" y="317" fontSize="5" fill="#94a3b8" className="italic px-1">FiLM Conditioning</text>

              <path d="M80 340 V355" stroke="#94a3b8" strokeWidth="0.8" markerEnd="url(#arrow)" />
              <text x="80" y="372" textAnchor="middle" fontSize="6.5" fill="#475569" className="font-medium">Dehazed Output</text>

              {/* Processing Animations */}
              {isProcessing && (
                <>
                  <path d="M80 50 V75" stroke="#3b82f6" strokeWidth="1.5" className="animate-flow" />
                  <path d="M80 130 V155" stroke="#3b82f6" strokeWidth="1.5" className="animate-flow" />
                  <path d="M80 245 V265" stroke="#3b82f6" strokeWidth="1.5" className="animate-flow" />
                </>
              )}
            </svg>
          </div>
        </section>

        {/* PANEL 2: UPLOAD INPUT */}
        <section className="lg:col-span-4 panel-container p-6 bg-[#f8fafc] flex flex-col">
          <div className="w-full flex items-center gap-2 mb-8 px-1">
            <svg className="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
            <h2 className="text-sm font-bold text-slate-700 tracking-tight">Upload Input</h2>
          </div>

          <div className="flex-1 flex flex-col gap-6">
            <div className="flex-1 bg-white border border-[#e2e8f0] rounded-sm p-8 flex flex-col items-center justify-center">
              <ImageUploader onImageSelect={handleImageSelect} preview={previewUrl} fileName={selectedImage?.name} />
            </div>

            <div className="flex flex-col gap-4">
              <button
                disabled={!selectedImage || isProcessing}
                onClick={processImage}
                className={`w-full py-3 rounded-sm text-sm font-bold tracking-wide transition-all ${selectedImage && !isProcessing
                    ? 'btn-primary'
                    : 'bg-[#94a3b8] text-white opacity-50 cursor-not-allowed'
                  }`}
              >
                {isProcessing ? 'PROCESING...' : 'Process Image'}
              </button>

              <div className="flex items-center gap-2 px-1">
                <span className="text-[10px] text-slate-400 font-bold uppercase">File:</span>
                <span className="text-[10px] text-slate-600 font-medium truncate">
                  {selectedImage ? selectedImage.name : 'no file selected'}
                </span>
              </div>
            </div>

            {error && (
              <div className="p-3 bg-red-50 text-red-500 border border-red-100 rounded-sm text-[10px] font-bold">
                ⚠ {error}
              </div>
            )}
          </div>
        </section>

        {/* PANEL 3: RESULTS (Stacked) */}
        <section className="lg:col-span-4 flex flex-col gap-6">
          {/* Input Preview */}
          <div className="panel-container p-6 bg-[#f8fafc] flex flex-col h-[340px]">
            <div className="flex justify-between items-center mb-6 px-1">
              <h2 className="text-sm font-bold text-slate-700">Input (Hazy)</h2>
              <svg className="w-3.5 h-3.5 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
            </div>
            <div className="flex-1 bg-white border border-[#e2e8f0] rounded-sm overflow-hidden relative group p-3">
              {previewUrl ? (
                <>
                  <div className="w-full h-full border border-slate-100 rounded-sm overflow-hidden relative">
                    <img src={previewUrl} className="w-full h-full object-cover" alt="Hazy Input" />
                    <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className="bg-[#1e293b]/70 text-white text-[9px] font-bold px-2 py-1.5 rounded-sm flex items-center gap-1.5 backdrop-blur-sm">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                        Download
                      </button>
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-0 right-0 text-center">
                    <span className="text-[10px] text-slate-400 font-bold bg-white/80 px-4 py-1">Input (Hazy)</span>
                  </div>
                </>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-200 text-[10px] font-bold uppercase tracking-widest">Awaiting Capture</div>
              )}
            </div>
          </div>

          {/* Output Preview */}
          <div className="panel-container p-6 bg-[#f8fafc] flex flex-col h-[340px]">
            <div className="flex justify-between items-center mb-6 px-1">
              <h2 className="text-sm font-bold text-slate-700">Output (Dehazed)</h2>
              <svg className="w-3.5 h-3.5 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
            </div>
            <div className="flex-1 bg-white border border-[#e2e8f0] rounded-sm overflow-hidden relative group p-3">
              {result ? (
                <>
                  <div className="w-full h-full border border-slate-100 rounded-sm overflow-hidden relative">
                    <img src={result.dehazed} className="w-full h-full object-cover" alt="Dehazed Output" />
                    <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <a href={result.dehazed} download="dehazed_result.png" className="bg-[#1e293b]/70 text-white text-[9px] font-bold px-2 py-1.5 rounded-sm flex items-center gap-1.5 backdrop-blur-sm">
                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" /></svg>
                        Download
                      </a>
                    </div>
                  </div>
                  <div className="absolute bottom-4 left-0 right-0 text-center">
                    <span className="text-[10px] text-slate-400 font-bold bg-white/80 px-4 py-1">Output (Dehazed)</span>
                  </div>
                </>
              ) : isProcessing ? (
                <div className="w-full h-full flex flex-col items-center justify-center text-slate-300 gap-4">
                  <div className="w-8 h-8 border-[3px] border-slate-100 border-t-blue-500 rounded-full animate-spin"></div>
                  <span className="text-[9px] font-bold tracking-[0.2em] text-slate-400">PROCESSING NODE...</span>
                </div>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-slate-200 text-[10px] font-bold uppercase tracking-widest">Buffer empty</div>
              )}
            </div>
          </div>
        </section>
      </div>

      <footer className="mt-14 text-[11px] text-[#64748b] font-medium tracking-tight bg-white/40 px-6 py-2 rounded-full border border-white/50 backdrop-blur-sm">
        Model: ResNet-18 (ImageNet) + Conditional U-Net | Kaggle paired dehazing set
      </footer>

      {isProcessing && <ProcessingLoader />}
    </main>
  );
}
