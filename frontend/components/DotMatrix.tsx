'use client';

import React, { useEffect, useRef } from 'react';

export default function DotMatrix() {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const mouseRef = useRef({ x: -2000, y: -2000 });
    const easedMouseRef = useRef({ x: -2000, y: -2000 });

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationFrameId: number;
        let width: number;
        let height: number;

        const handleResize = () => {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width * window.devicePixelRatio;
            canvas.height = height * window.devicePixelRatio;
            ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        };

        const handleMouseMove = (e: MouseEvent) => {
            mouseRef.current = { x: e.clientX, y: e.clientY };
        };

        const draw = () => {
            ctx.clearRect(0, 0, width, height);

            // Ultra-fast easing for precision
            easedMouseRef.current.x += (mouseRef.current.x - easedMouseRef.current.x) * 0.15;
            easedMouseRef.current.y += (mouseRef.current.y - easedMouseRef.current.y) * 0.15;

            const spacing = 32; // Wider grid for blueprint feel
            const dotSize = 1;
            const radius = 120; // Tight influence area

            for (let x = 0; x < width + spacing; x += spacing) {
                for (let y = 0; y < height + spacing; y += spacing) {
                    const dx = x - easedMouseRef.current.x;
                    const dy = y - easedMouseRef.current.y;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    if (dist < radius) {
                        const opacity = (1 - dist / radius) * 0.15; // Extremely subtle
                        ctx.fillStyle = `rgba(59, 130, 246, ${opacity})`;
                        ctx.beginPath();
                        ctx.arc(x, y, dotSize, 0, Math.PI * 2);
                        ctx.fill();
                    }
                }
            }

            animationFrameId = requestAnimationFrame(draw);
        };

        window.addEventListener('resize', handleResize);
        window.addEventListener('mousemove', handleMouseMove);

        handleResize();
        draw();

        return () => {
            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
            cancelAnimationFrame(animationFrameId);
        };
    }, []);

    return (
        <canvas
            ref={canvasRef}
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: -1,
                pointerEvents: 'none',
            }}
        />
    );
}
