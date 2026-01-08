'use client';

import React from 'react';

interface ImageUploaderProps {
    onImageSelect: (file: File) => void;
    preview: string | null;
    fileName?: string;
}

export default function ImageUploader({ onImageSelect, preview, fileName }: ImageUploaderProps) {
    const onFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) onImageSelect(file);
    };

    return (
        <div className="w-full flex flex-col gap-6">
            {/* Upload Area */}
            <div className="w-full h-40 border-2 border-dashed border-slate-200 rounded-lg flex flex-col items-center justify-center p-4 bg-white/50 relative">
                <input type="file" className="hidden" id="file-upload" accept="image/*" onChange={onFileSelect} />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-3">
                    <div className="text-slate-400">
                        <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                        </svg>
                    </div>
                    <div className="text-center">
                        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest leading-tight">Drag & Drop Image File Here</p>
                        <p className="text-[9px] text-slate-400 uppercase tracking-widest mt-1">Or Click to Browse</p>
                    </div>
                </label>
            </div>

            {/* Buttons */}
            <div className="flex flex-col gap-2">
                <label className="cursor-pointer w-full py-2 px-4 bg-slate-50 border border-slate-100 rounded text-[10px] font-semibold text-slate-600 flex items-center justify-center gap-2 hover:bg-slate-100 transition-colors">
                    <input type="file" className="hidden" accept="image/*" onChange={onFileSelect} />
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                    Upload Image
                </label>

                <button disabled className="w-full py-2 px-4 bg-slate-50 border border-slate-100 rounded text-[10px] font-semibold text-slate-300 flex items-center justify-center gap-2 cursor-not-allowed">
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                    Upload Video (Work in Progress)
                </button>
            </div>
        </div>
    );
}
