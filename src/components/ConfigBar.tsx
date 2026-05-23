"use client";

import { useState, useEffect } from "react";
import { useGraphStore } from "@/store/graphStore";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Settings, Key, Cpu, RotateCcw, Target, ChevronDown, ChevronUp, CheckCircle2, Zap } from "lucide-react";

export function ConfigBar() {
  const { config, setConfig } = useGraphStore();
  const [expanded, setExpanded] = useState(false);
  const [hasServerKey, setHasServerKey] = useState(false);

  // Check if server-side OPENCODE_API_KEY is available
  useEffect(() => {
    fetch("/api/config")
      .then((res) => res.json())
      .then((data) => setHasServerKey(data.hasServerKey))
      .catch(() => {});
  }, []);

  return (
    <div className="bg-[#12122a] border-b border-[#2a2a5a]">
      {/* Always-visible row: Logo + API Key + Model + Expand toggle */}
      <div className="flex items-center gap-3 px-3 py-2">
        {/* Logo */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="w-6 h-6 rounded-md bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
            <Settings className="w-3.5 h-3.5 text-white" />
          </div>
          <span className="text-[#e0e0ff] font-mono font-bold text-xs tracking-wide hidden sm:inline">
            Atomic Graph
          </span>
        </div>

        {/* API Key — always visible */}
        <div className="flex items-center gap-1.5 flex-1 min-w-0">
          <Key className="w-3.5 h-3.5 text-[#8888cc] shrink-0" />
          <Input
            type="password"
            placeholder="OpenCode API Key (optional if set on server)"
            value={config.apiKey}
            onChange={(e) => setConfig({ apiKey: e.target.value })}
            className="h-7 bg-[#1a1a3e] border-[#3a3a6a] text-[#e0e0ff] placeholder-[#6666aa] text-[11px] font-mono focus:border-indigo-500 focus:ring-indigo-500/30"
          />
          {hasServerKey && !config.apiKey && (
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" title="Server key available" />
          )}
        </div>

        {/* Model badge — compact */}
        <Badge
          variant="secondary"
          className="bg-[#1a1a3e] text-[#c8c8ee] font-mono text-[10px] border border-[#3a3a6a] px-2 py-0.5 shrink-0 hidden sm:flex"
        >
          <Cpu className="w-3 h-3 mr-1" />
          GLM 5.1
        </Badge>

        {/* Mobile: expand/collapse toggle */}
        <button
          onClick={() => setExpanded(!expanded)}
          className="md:hidden flex items-center justify-center w-7 h-7 rounded bg-[#1a1a3e] border border-[#3a3a6a] text-[#8888cc]"
        >
          {expanded ? (
            <ChevronUp className="w-3.5 h-3.5" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5" />
          )}
        </button>
      </div>

      {/* Expandable settings row — visible on desktop always, on mobile when expanded */}
      <div
        className={`${
          expanded ? "flex" : "hidden md:flex"
        } flex-wrap items-center gap-3 px-3 pb-2`}
      >
        {/* Model badge — mobile only (shown in expanded) */}
        <div className="flex items-center gap-1.5 sm:hidden">
          <Cpu className="w-3.5 h-3.5 text-[#8888cc] shrink-0" />
          <Badge
            variant="secondary"
            className="bg-[#1a1a3e] text-[#c8c8ee] font-mono text-[10px] border border-[#3a3a6a] px-2 py-0.5"
          >
            GLM 5.1
          </Badge>
        </div>

        {/* Fast Mode Toggle */}
        <div className="flex items-center gap-1.5">
          <Zap className={`w-3.5 h-3.5 shrink-0 ${config.fastMode ? "text-amber-400" : "text-[#8888cc]"}`} />
          <button
            onClick={() => setConfig({ fastMode: !config.fastMode })}
            className={`flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-mono border transition-colors ${
              config.fastMode
                ? "bg-amber-500/20 border-amber-500/40 text-amber-300"
                : "bg-[#1a1a3e] border-[#3a3a6a] text-[#8888cc] hover:text-[#c8c8ee]"
            }`}
          >
            Fast
          </button>
        </div>

        {/* Iterations Slider */}
        <div className="flex items-center gap-1.5">
          <RotateCcw className="w-3.5 h-3.5 text-[#8888cc] shrink-0" />
          <Label className="text-[#aaaadd] text-[11px] font-mono whitespace-nowrap">
            Iter: {config.iterations}
          </Label>
          <Slider
            min={1}
            max={5}
            step={1}
            value={[config.iterations]}
            onValueChange={([val]) => setConfig({ iterations: val })}
            className="w-[70px]"
          />
        </div>

        {/* Confidence Threshold */}
        <div className="flex items-center gap-1.5">
          <Target className="w-3.5 h-3.5 text-[#8888cc] shrink-0" />
          <Label className="text-[#aaaadd] text-[11px] font-mono whitespace-nowrap">
            Threshold: {config.confidenceThreshold.toFixed(2)}
          </Label>
          <Input
            type="number"
            min={0}
            max={1}
            step={0.05}
            value={config.confidenceThreshold}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!isNaN(val) && val >= 0 && val <= 1) {
                setConfig({ confidenceThreshold: val });
              }
            }}
            className="h-7 w-[60px] bg-[#1a1a3e] border-[#3a3a6a] text-[#e0e0ff] text-[11px] font-mono"
          />
        </div>
      </div>
    </div>
  );
}
