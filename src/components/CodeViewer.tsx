import React, { useState } from "react";
import { CODEBASE_FILES, CodeFile } from "../data/codebase";
import {
  FileCode,
  Copy,
  Check,
  Download,
  Terminal,
  FolderArchive,
  ExternalLink,
  Layers,
} from "lucide-react";

export const CodeViewer: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<CodeFile>(CODEBASE_FILES[1]); // tracker.py default
  const [copied, setCopied] = useState<boolean>(false);
  const [downloadAllStatus, setDownloadAllStatus] = useState<string | null>(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(selectedFile.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadSingle = (file: CodeFile) => {
    const blob = new Blob([file.content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = file.filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  const handleDownloadAll = () => {
    // Downloads all files sequentially
    CODEBASE_FILES.forEach((f, idx) => {
      setTimeout(() => {
        handleDownloadSingle(f);
      }, idx * 250);
    });
    setDownloadAllStatus("Downloaded all project files!");
    setTimeout(() => setDownloadAllStatus(null), 3000);
  };

  const lines = selectedFile.content.split("\n");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-[750px]">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-3 bg-slate-950 border-b border-slate-800 gap-3">
        <div className="flex items-center gap-2">
          <Terminal className="w-5 h-5 text-cyan-400" />
          <span className="font-semibold text-slate-100 text-sm">
            Python Project Codebase Explorer
          </span>
          <span className="text-xs px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/40 text-cyan-300 font-mono">
            FastAPI + YOLOv11 + ByteTrack
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-emerald-400" />
                <span className="text-emerald-400">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5 text-slate-400" />
                <span>Copy File</span>
              </>
            )}
          </button>

          <button
            onClick={() => handleDownloadSingle(selectedFile)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-mono transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Download {selectedFile.filename}</span>
          </button>

          <button
            onClick={handleDownloadAll}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs font-mono transition-colors shadow-sm"
          >
            <FolderArchive className="w-3.5 h-3.5" />
            <span>Download All Files</span>
          </button>
        </div>
      </div>

      {downloadAllStatus && (
        <div className="px-4 py-2 bg-emerald-950/60 border-b border-emerald-800/50 text-emerald-300 text-xs font-mono flex items-center justify-between">
          <span>{downloadAllStatus}</span>
          <span className="text-[11px] text-emerald-400/80">6 files bundled</span>
        </div>
      )}

      {/* Main split: File Sidebar + Code View */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left File List */}
        <div className="w-64 border-r border-slate-800 bg-slate-950/50 p-3 space-y-1 overflow-y-auto">
          <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider px-2 py-1 flex items-center gap-1.5 font-mono">
            <Layers className="w-3.5 h-3.5" />
            Project Files
          </div>

          {CODEBASE_FILES.map((file) => {
            const isSelected = selectedFile.filename === file.filename;
            return (
              <button
                key={file.filename}
                onClick={() => setSelectedFile(file)}
                className={`w-full text-left px-3 py-2.5 rounded-lg text-xs font-mono transition-all flex items-start gap-2.5 ${
                  isSelected
                    ? "bg-cyan-500/15 text-cyan-300 border border-cyan-500/30"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
                }`}
              >
                <FileCode
                  className={`w-4 h-4 mt-0.5 shrink-0 ${
                    isSelected ? "text-cyan-400" : "text-slate-500"
                  }`}
                />
                <div>
                  <div className="font-medium text-slate-200">{file.filename}</div>
                  <div className="text-[10px] text-slate-500 line-clamp-1 mt-0.5">
                    {file.language.toUpperCase()}
                  </div>
                </div>
              </button>
            );
          })}

          <div className="pt-4 mt-4 border-t border-slate-800/80 px-2">
            <div className="text-[11px] font-semibold text-slate-400 mb-2 font-mono">
              Quick Run Instructions:
            </div>
            <div className="bg-slate-950 p-2.5 rounded border border-slate-800 text-[10px] font-mono text-slate-400 space-y-1">
              <p className="text-cyan-300">$ pip install -r requirements.txt</p>
              <p className="text-emerald-300">$ uvicorn main:app --reload</p>
              <p className="text-amber-300">$ python test_client.py</p>
            </div>
          </div>
        </div>

        {/* Right Code Display */}
        <div className="flex-1 flex flex-col overflow-hidden bg-slate-950">
          {/* File description header */}
          <div className="px-4 py-2.5 bg-slate-900/60 border-b border-slate-800/80 text-xs text-slate-400 font-mono flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="text-cyan-400 font-bold">{selectedFile.filename}</span>
              <span>— {selectedFile.description}</span>
            </span>
            <span className="text-slate-500">{lines.length} lines</span>
          </div>

          {/* Scrollable code with line numbers */}
          <div className="flex-1 overflow-auto p-4 font-mono text-xs leading-relaxed">
            <pre className="flex text-slate-300">
              {/* Line numbers column */}
              <div className="select-none text-slate-600 text-right pr-4 border-r border-slate-800/60 mr-4 shrink-0">
                {lines.map((_, i) => (
                  <div key={i}>{i + 1}</div>
                ))}
              </div>
              {/* Actual code content */}
              <code className="text-slate-200 whitespace-pre overflow-x-auto">
                {selectedFile.content}
              </code>
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
};
