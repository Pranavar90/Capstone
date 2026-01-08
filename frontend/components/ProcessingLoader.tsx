'use client';

import React from 'react';

export default function ProcessingLoader() {
    return (
        <div className="fixed inset-0 z-[500] flex items-center justify-center bg-slate-900/10 backdrop-blur-sm animate-fade-in pointer-events-none">
            <div className="bg-white p-8 rounded-2xl shadow-2xl flex flex-col items-center gap-6 max-w-xs text-center border border-slate-100">
                <div className="relative w-16 h-16">
                    <div className="absolute inset-0 border-4 border-slate-100 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                </div>

                <div className="space-y-2">
                    <h2 className="text-sm font-bold text-slate-900 tracking-tight">Applying Neural Weights</h2>
                    <p className="text-[10px] text-slate-500 font-medium leading-relaxed px-4">
                        Calculating transmission maps and recovery radiance on CUDA node.
                    </p>
                </div>

                <div className="w-full flex justify-between items-center px-4">
                    <div className="h-1 flex-1 bg-slate-100 rounded-full overflow-hidden mr-3">
                        <div className="h-full bg-blue-500 w-2/3 animate-[pulse_1.5s_infinite]"></div>
                    </div>
                    <span className="text-[9px] font-bold text-blue-600 mono">RUNNING</span>
                </div>
            </div>
        </div>
    );
}
