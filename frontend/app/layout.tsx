import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Weather-Adaptive Image Dehazing | AI-Powered Dehazing System",
  description: "CUDA-accelerated image dehazing using conditional U-Net with ResNet-18 feature extraction. Transform hazy images into clear, vibrant photos with AI.",
  keywords: ["image dehazing", "AI", "PyTorch", "CUDA", "computer vision", "weather-adaptive"],
  authors: [{ name: "Dehazing System" }],
  openGraph: {
    title: "Weather-Adaptive Image Dehazing",
    description: "AI-powered image dehazing with CUDA acceleration",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className={inter.className}>{children}</body>
    </html>
  );
}
