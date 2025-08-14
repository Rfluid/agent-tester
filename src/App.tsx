import { useEffect, useRef, useState } from "react";
import "./index.css";

// ---- Shareable session helpers ----

// URL-safe base64 (no padding, - _ instead of + /)
const toUrlB64 = (s: string) =>
    btoa(encodeURIComponent(s).replace(/%([0-9A-F]{2})/g, (_, p1) => String.fromCharCode(parseInt(p1, 16))))
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=+$/g, "");

const fromUrlB64 = (b64: string) => {
    const pad = b64.length % 4 === 0 ? "" : "=".repeat(4 - (b64.length % 4));
    const norm = b64.replace(/-/g, "+").replace(/_/g, "/") + pad;
    const str = atob(norm);
    // decodeURIComponent for the earlier encodeURIComponent
    const escaped = Array.prototype.map
        .call(str, (c: string) => "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2))
        .join("");
    return decodeURIComponent(escaped);
};

async function copyToClipboard(text: string) {
    // Try modern API
    try {
        if (navigator.clipboard && window.isSecureContext) {
            await navigator.clipboard.writeText(text);
            return true;
        }
    } catch (e) {
        console.error(e);
    }
    // Fallback via hidden textarea
    try {
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        ta.style.top = "0";
        ta.setAttribute("readonly", "true");
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        if (ok) return true;
    } catch (e) {
        console.error(e);
    }
    // Last resort: prompt (lets user Cmd/Ctrl+C)
    window.prompt("Copy this link:", text);
    return false;
}

// --- Types coming from your agent contract ---

type WebSocketData = "delta" | "final";

interface BaseLLMResponse {
    action_payloads?: unknown | null;
    next_step?: "end";
    next_step_reason?: string;
}

interface LLMAPIResponse extends BaseLLMResponse {
    response?: string;
}

interface LLMWebSocketResponse {
    type: WebSocketData;
    data: Partial<LLMAPIResponse> | null;
}

// --- UI types ---

type Role = "user" | "agent";

interface ConversationItem {
    id: string;
    role: Role;
    text: string;
    at: string; // ISO timestamp with milliseconds
    // Optional extra fields from agent response
    extra?: Partial<BaseLLMResponse> | null;
}

interface TimingRecord {
    id: string; // same id used for the agent turn
    requestedAt: string; // when we asked agent to generate
    firstDeltaAt?: string; // first delta arrival time
    finalAt?: string; // final message arrival time
}

// Utilities
const isoNow = () => new Date().toISOString(); // includes ms
const uuid = () => crypto.randomUUID();

// Persist small prefs in localStorage
const useLocalStorage = <T,>(key: string, initial: T) => {
    const [value, setValue] = useState<T>(() => {
        const raw = localStorage.getItem(key);
        return raw ? (JSON.parse(raw) as T) : initial;
    });
    useEffect(() => {
        localStorage.setItem(key, JSON.stringify(value));
    }, [key, value]);
    return [value, setValue] as const;
};

// --- Main App ---
export default function App() {
    const [agentUrl, setAgentUrl] = useLocalStorage("agentUrl", "ws://localhost:8000/agent/ws");
    const [openaiKey, setOpenaiKey] = useLocalStorage("openaiKey", ""); // NOTE: never shared
    const [threadId, setThreadId] = useLocalStorage("threadId", "test-thread-001");
    const [maxRetries, setMaxRetries] = useLocalStorage("maxRetries", 3);
    const [loopThreshold, setLoopThreshold] = useLocalStorage("loopThreshold", 5);
    const [topK, setTopK] = useLocalStorage("topK", 5);

    const [numTurns, setNumTurns] = useLocalStorage("numTurns", 3);
    const [turnDelayMs, setTurnDelayMs] = useLocalStorage("turnDelayMs", 0);

    const [wsConnected, setWsConnected] = useState(false);
    const [wsError, setWsError] = useState<string | null>(null);
    const wsRef = useRef<WebSocket | null>(null);

    // Conversation state
    const [conversation, setConversation] = useState<ConversationItem[]>([]);
    const conversationRef = useRef<ConversationItem[]>(conversation);
    useEffect(() => {
        conversationRef.current = conversation;
    }, [conversation]);

    const [timings, setTimings] = useState<Record<string, TimingRecord>>({});

    const [mockPrompt, setMockPrompt] = useLocalStorage(
        "mockPrompt",
        "Generate the FIRST short, natural message from a user starting a conversation with an agent. Do not include explanations.",
    );
    const [openaiModel, setOpenaiModel] = useLocalStorage("openaiModel", "gpt-4o-mini");
    const [isGenerating, setIsGenerating] = useState(false);

    // Auto-run state
    const [isRunning, setIsRunning] = useState(false);
    const stopRequestedRef = useRef(false);
    const finalResolversRef = useRef(new Map<string, () => void>()); // turnId -> resolve()

    // In-flight turn pointer (ensures all deltas land on the same turnId)
    const activeTurnIdRef = useRef<string | null>(null);

    // progress, stop loading, and user-synthesis tracking
    const [completedTurns, setCompletedTurns] = useState(0);
    const [isStopping, setIsStopping] = useState(false);
    const [isSynthesizingUser, setIsSynthesizingUser] = useState(false);
    const isSynthUserRef = useRef(false);
    useEffect(() => {
        isSynthUserRef.current = isSynthesizingUser;
    }, [isSynthesizingUser]);

    // simple toast for copied links / imports
    const [toast, setToast] = useState<string | null>(null);
    useEffect(() => {
        if (!toast) return;
        const t = setTimeout(() => setToast(null), 2000);
        return () => clearTimeout(t);
    }, [toast]);

    const scrollRef = useRef<HTMLDivElement | null>(null);
    useEffect(() => {
        scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
    }, [conversation]);

    // WebSocket connect/disconnect
    const connect = () => {
        try {
            setWsError(null);
            const socket = new WebSocket(agentUrl);
            wsRef.current = socket;
            socket.onopen = () => setWsConnected(true);
            socket.onclose = () => setWsConnected(false);
            socket.onerror = () => setWsError("WebSocket error – check agent URL and CORS");
            socket.onmessage = (ev) => {
                try {
                    const payload = JSON.parse(ev.data) as LLMWebSocketResponse;
                    handleAgentEvent(payload);
                } catch {
                    console.error("Invalid WS message:", ev.data);
                }
            };
        } catch (e: unknown) {
            setWsError(String(e));
        }
    };

    const disconnect = () => {
        wsRef.current?.close();
        wsRef.current = null;
    };

    // Handle incoming agent events (always uses the active turn; deltas do not create new bubbles)
    const handleAgentEvent = (evt: LLMWebSocketResponse) => {
        const { type, data } = evt;

        const turnId = activeTurnIdRef.current;
        if (!turnId) {
            console.warn("Delta/final received without active turn. Event ignored:", evt);
            return;
        }

        appendAgentDelta(turnId, type, data);

        if (type === "final") {
            activeTurnIdRef.current = null;
        }
    };

    const createTimingRecord = (id: string) => {
        setTimings((prev) => ({
            ...prev,
            [id]: { id, requestedAt: isoNow() },
        }));
    };

    const appendAgentDelta = (turnId: string, type: WebSocketData, data: Partial<LLMAPIResponse> | null) => {
        setTimings((prev) => {
            const t = prev[turnId];
            if (!t) return prev;
            if (type === "delta" && !t.firstDeltaAt) t.firstDeltaAt = isoNow();
            if (type === "final") t.finalAt = isoNow();
            return { ...prev, [turnId]: { ...t } };
        });

        setConversation((prev) => {
            const idx = prev.findIndex((m) => m.id === turnId && m.role === "agent");
            const existing = idx >= 0 ? prev[idx] : null;
            const nextText = data?.response ?? "";
            const updated: ConversationItem = {
                id: turnId,
                role: "agent",
                text: nextText,
                at: isoNow(),
                extra: data
                    ? {
                          action_payloads: data.action_payloads,
                          next_step: data.next_step,
                          next_step_reason: data.next_step_reason,
                      }
                    : existing?.extra,
            };

            const copy = [...prev];
            if (existing) copy[idx] = updated;
            else copy.push(updated);
            return copy;
        });

        if (type === "final") {
            const resolver = finalResolversRef.current.get(turnId);
            if (resolver) {
                finalResolversRef.current.delete(turnId);
                resolver();
            }
            setCompletedTurns((n) => n + 1);
        }
    };

    // --- OpenAI helpers ---
    function extractTextFromResponsesPayload(json): string | null {
        if (typeof json?.output_text === "string" && json.output_text.trim()) return json.output_text.trim();
        const out = json?.output;
        if (Array.isArray(out)) {
            for (const item of out) {
                if (item?.type === "message" && Array.isArray(item?.content)) {
                    for (const c of item.content) {
                        if (c?.type === "output_text" && typeof c?.text === "string" && c.text.trim()) {
                            return c.text.trim();
                        }
                    }
                }
            }
        }
        const choices0 = json?.choices?.[0]?.message?.content;
        if (typeof choices0 === "string" && choices0.trim()) return choices0.trim();
        const legacy = json?.output?.[0]?.content?.[0]?.text;
        if (typeof legacy === "string" && legacy.trim()) return legacy.trim();
        return null;
    }

    async function generateFirstUserMessage({
        apiKey,
        model,
        prompt,
    }: {
        apiKey: string;
        model: string;
        prompt: string;
    }): Promise<string> {
        return generateNextUserMessage({ apiKey, model, conversation: [], basePrompt: prompt });
    }

    async function generateNextUserMessage({
        apiKey,
        model,
        conversation,
        basePrompt,
    }: {
        apiKey: string;
        model: string;
        conversation: ConversationItem[];
        basePrompt: string;
    }): Promise<string> {
        const messages = [
            {
                role: "system",
                content:
                    "You are a generator that interprets the history and produces the NEXT utterance from the human USER, short, natural, and contextual. Do not explain; only write the user's sentence. Keep the language and tone of the history.",
            },
            {
                role: "user",
                content:
                    `${basePrompt}\n\nBelow is the history in the format 'User:' and 'Agent:'. Write the NEXT User message in 5 to 25 words, keeping coherence.\n\n` +
                    conversation.map((m) => (m.role === "user" ? `User: ${m.text}` : `Agent: ${m.text}`)).join("\n"),
            },
        ];

        try {
            const res = await fetch("https://api.openai.com/v1/responses", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${apiKey}`,
                },
                body: JSON.stringify({ model, input: messages }),
            });
            if (!res.ok) throw new Error(`Responses API HTTP ${res.status}`);
            const json = await res.json();
            const text = extractTextFromResponsesPayload(json);
            if (typeof text === "string" && text.trim()) return text.trim();
            throw new Error("Unexpected Responses API payload shape");
        } catch {
            const res2 = await fetch("https://api.openai.com/v1/chat/completions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Authorization: `Bearer ${apiKey}`,
                },
                body: JSON.stringify({ model, messages, temperature: 0.7 }),
            });
            if (!res2.ok) throw new Error(`Chat Completions HTTP ${res2.status}`);
            const json2 = await res2.json();
            const text2 = json2?.choices?.[0]?.message?.content;
            if (typeof text2 === "string" && text2.trim()) return text2.trim();
            throw new Error("Unexpected Chat Completions payload shape");
        }
    }

    // --- Turn runner ---
    const waitForFinal = (turnId: string) =>
        new Promise<void>((resolve) => {
            finalResolversRef.current.set(turnId, resolve);
        });

    const sendToAgent = (userMsg: string, agentTurnId: string) => {
        const payload = {
            data: userMsg,
            chat_interface: "websocket" as const,
            max_retries: Math.max(0, Number(maxRetries) || 0),
            loop_threshold: Math.max(1, Number(loopThreshold) || 1),
            top_k: Math.max(0, Number(topK) || 0),
            thread_id: threadId,
            _client_turn_id: agentTurnId,
        };

        activeTurnIdRef.current = agentTurnId;
        wsRef.current?.send(JSON.stringify(payload));
        createTimingRecord(agentTurnId);
    };

    // wait-until-idle helper used by Stop
    const waitUntilIdle = async () => {
        const currentTurn = activeTurnIdRef.current;
        if (currentTurn) {
            try {
                await waitForFinal(currentTurn);
            } catch (e) {
                console.error(e);
            }
        }
        while (isSynthUserRef.current) {
            await new Promise((r) => setTimeout(r, 50));
        }
    };

    const runTurns = async (turns: number) => {
        if (!openaiKey) {
            alert("Provide your OpenAI API key to generate mock user messages.");
            return;
        }
        if (!wsConnected) {
            connect();
            await new Promise((r) => setTimeout(r, 250));
        }

        setIsRunning(true);
        stopRequestedRef.current = false;

        try {
            if (
                conversationRef.current.length === 0 ||
                conversationRef.current[conversationRef.current.length - 1].role === "agent"
            ) {
                if (stopRequestedRef.current) throw new Error("Stopped");
                setIsSynthesizingUser(true);
                const firstMsg = await generateFirstUserMessage({
                    apiKey: openaiKey,
                    model: openaiModel,
                    prompt: mockPrompt,
                });
                setIsSynthesizingUser(false);

                const userTurnId = uuid();
                setConversation((prev) => [...prev, { id: userTurnId, role: "user", text: firstMsg, at: isoNow() }]);

                const agentTurnId = uuid();
                sendToAgent(firstMsg, agentTurnId);
                await waitForFinal(agentTurnId);
                if (turnDelayMs > 0) await new Promise((r) => setTimeout(r, turnDelayMs));
            }

            let remaining = Math.max(0, turns - 1);
            while (remaining-- > 0) {
                if (stopRequestedRef.current) break;
                setIsSynthesizingUser(true);
                const nextMsg = await generateNextUserMessage({
                    apiKey: openaiKey,
                    model: openaiModel,
                    conversation: conversationRef.current,
                    basePrompt: mockPrompt,
                });
                setIsSynthesizingUser(false);

                const userTurnId = uuid();
                setConversation((prev) => [...prev, { id: userTurnId, role: "user", text: nextMsg, at: isoNow() }]);

                const agentTurnId = uuid();
                sendToAgent(nextMsg, agentTurnId);
                await waitForFinal(agentTurnId);
                if (turnDelayMs > 0) await new Promise((r) => setTimeout(r, turnDelayMs));
            }
        } catch (e) {
            console.warn(e);
        } finally {
            setIsRunning(false);
            stopRequestedRef.current = false;
        }
    };

    // Single-run
    const startTest = async () => {
        if (!wsConnected) {
            connect();
            await new Promise((r) => setTimeout(r, 250));
        }
        if (!openaiKey) {
            alert("Provide your OpenAI API key to generate the initial message.");
            return;
        }
        setIsGenerating(true);
        setIsSynthesizingUser(true);
        try {
            const userMsg = await generateFirstUserMessage({
                apiKey: openaiKey,
                model: openaiModel,
                prompt: mockPrompt,
            });
            const userTurnId = uuid();
            setConversation((prev) => [...prev, { id: userTurnId, role: "user", text: userMsg, at: isoNow() }]);
            const agentTurnId = uuid();
            sendToAgent(userMsg, agentTurnId);
        } catch (e: unknown) {
            console.error(e);
            alert("Failed to generate initial message with the OpenAI API. Check console.");
        } finally {
            setIsSynthesizingUser(false);
            setIsGenerating(false);
        }
    };

    // Stop & Continue
    const handleStop = async () => {
        setIsStopping(true);
        stopRequestedRef.current = true;
        try {
            await waitUntilIdle();
        } finally {
            setIsStopping(false);
            setIsRunning(false);
        }
    };

    const handleContinue = async () => {
        const remaining = Math.max(0, Number(numTurns) - Number(completedTurns));
        if (remaining <= 0) return;
        await runTurns(remaining);
    };

    function serializeSession() {
        return {
            v: 1,
            config: {
                agentUrl,
                threadId,
                maxRetries,
                loopThreshold,
                topK,
                numTurns,
                turnDelayMs,
                mockPrompt,
                openaiModel,
                // DO NOT include openaiKey
            },
            conversation,
            timings,
            completedTurns,
        };
    }

    function applySession(s) {
        if (!s || typeof s !== "object") return;
        const { config, conversation: conv, timings: tmg, completedTurns: ct } = s;
        if (config) {
            setAgentUrl(config.agentUrl ?? agentUrl);
            setThreadId(config.threadId ?? threadId);
            setMaxRetries(config.maxRetries ?? maxRetries);
            setLoopThreshold(config.loopThreshold ?? loopThreshold);
            setTopK(config.topK ?? topK);
            setNumTurns(config.numTurns ?? numTurns);
            setTurnDelayMs(config.turnDelayMs ?? turnDelayMs);
            setMockPrompt(config.mockPrompt ?? mockPrompt);
            setOpenaiModel(config.openaiModel ?? openaiModel);
        }
        if (Array.isArray(conv)) setConversation(conv);
        if (tmg && typeof tmg === "object") setTimings(tmg);
        if (typeof ct === "number") setCompletedTurns(ct);
    }

    const shareSession = async () => {
        try {
            const payload = serializeSession();
            const json = JSON.stringify(payload);
            const token = toUrlB64(json);
            const url = `${location.origin}${location.pathname}#s=${token}`;
            const ok = await copyToClipboard(url);
            if (!ok) alert("Link shown; copy it manually.");
        } catch (e) {
            console.error(e);
            alert("Failed to generate share link. See console.");
        }
    };

    const importSessionFromHash = () => {
        const { hash } = location;
        if (!hash.startsWith("#s=")) return false;
        try {
            const token = hash.slice(3);
            const json = fromUrlB64(token);
            const data = JSON.parse(json);
            applySession(data);
            return true;
        } catch (e) {
            console.warn("Invalid shared session in URL.", e);
            return false;
        }
    };

    useEffect(() => {
        if (importSessionFromHash()) {
            // Optional: notify user
            console.log("Imported session from link.");
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Clear conversation + timings and start a fresh session (new threadId)
    const clearAndNewSession = () => {
        // stop any running flows
        stopRequestedRef.current = true;
        activeTurnIdRef.current = null;

        setConversation([]);
        setTimings({});
        setCompletedTurns(0);

        // new thread id to force a cold context on the server
        const newTid = uuid();
        setThreadId(newTid);

        // remove share hash
        history.replaceState(null, "", location.pathname);

        // (optional) disconnect socket so the server sees a fresh start
        // disconnect(); // uncomment if you prefer a hard reset
    };

    const exportReport = () => {
        const blob = new Blob([JSON.stringify({ conversation, timings }, null, 2)], {
            type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `agent-test-${new Date().toISOString()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900">
            <div className="mx-auto px-4 py-6">
                <header className="mb-6 flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                        <h1 className="text-2xl font-semibold">Agent Tester</h1>
                        <p className="text-sm text-slate-600">
                            Generate messages via OpenAI, send them to your agent over WebSocket, monitor deltas/final,
                            and export/share sessions.
                        </p>
                        <p className="text-xs text-slate-600 mt-1">
                            Progress: <span className="font-mono">{completedTurns}</span>/
                            <span className="font-mono">{numTurns}</span> turns
                        </p>
                        {toast && (
                            <div className="mt-2 inline-block rounded-xl border px-3 py-1 text-xs bg-white shadow">
                                {toast}
                            </div>
                        )}
                    </div>
                    <div className="flex gap-3">
                        {!wsConnected ? (
                            <button
                                onClick={connect}
                                className="px-3 py-2 rounded-2xl shadow bg-white border hover:bg-slate-50"
                            >
                                Connect WS
                            </button>
                        ) : (
                            <button
                                onClick={disconnect}
                                className="px-3 py-2 rounded-2xl shadow bg-white border hover:bg-slate-50"
                            >
                                Disconnect WS
                            </button>
                        )}

                        <button
                            onClick={exportReport}
                            className="px-3 py-2 rounded-2xl shadow bg-white border hover:bg-slate-50"
                        >
                            Export JSON
                        </button>

                        <button
                            onClick={shareSession}
                            className="px-3 py-2 rounded-2xl shadow bg-white border hover:bg-slate-50"
                        >
                            Share link
                        </button>

                        <button
                            onClick={clearAndNewSession}
                            className="px-3 py-2 rounded-2xl shadow bg-white border hover:bg-slate-50"
                        >
                            Clear & New Session
                        </button>
                    </div>
                </header>

                {/* Config Form */}
                <section className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                    <div className="space-y-3 p-4 rounded-2xl bg-white shadow">
                        <h2 className="font-medium">Connection & Parameters</h2>
                        <label className="block text-sm">
                            Agent URL
                            <input
                                value={agentUrl}
                                onChange={(e) => setAgentUrl(e.target.value)}
                                placeholder="ws://localhost:8000/agent/ws"
                                className="mt-1 w-full rounded-xl border px-3 py-2"
                            />
                        </label>
                        <div className="grid grid-cols-3 gap-3">
                            <label className="block text-sm">
                                max_retries
                                <input
                                    type="number"
                                    min={0}
                                    value={maxRetries}
                                    onChange={(e) => setMaxRetries(Number(e.target.value))}
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                            <label className="block text-sm">
                                loop_threshold
                                <input
                                    type="number"
                                    min={1}
                                    value={loopThreshold}
                                    onChange={(e) => setLoopThreshold(Number(e.target.value))}
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                            <label className="block text-sm">
                                top_k
                                <input
                                    type="number"
                                    min={0}
                                    value={topK}
                                    onChange={(e) => setTopK(Number(e.target.value))}
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                        </div>
                        <label className="block text-sm">
                            Thread ID
                            <input
                                value={threadId}
                                onChange={(e) => setThreadId(e.target.value)}
                                placeholder="test-thread-001"
                                className="mt-1 w-full rounded-xl border px-3 py-2"
                            />
                        </label>
                        <div className="grid grid-cols-2 gap-3">
                            <label className="block text-sm">
                                Number of turns
                                <input
                                    type="number"
                                    min={1}
                                    value={numTurns}
                                    onChange={(e) => setNumTurns(Math.max(1, Number(e.target.value) || 1))}
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                            <label className="block text-sm">
                                Delay between turns (ms)
                                <input
                                    type="number"
                                    min={0}
                                    value={turnDelayMs}
                                    onChange={(e) => setTurnDelayMs(Math.max(0, Number(e.target.value) || 0))}
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                        </div>
                    </div>

                    <div className="space-y-3 p-4 rounded-2xl bg-white shadow">
                        <h2 className="font-medium">User Mock (OpenAI)</h2>
                        <label className="block text-sm">
                            OpenAI API Key
                            <input
                                value={openaiKey}
                                onChange={(e) => setOpenaiKey(e.target.value)}
                                placeholder="sk-..."
                                className="mt-1 w-full rounded-xl border px-3 py-2"
                            />
                        </label>
                        <div className="grid grid-cols-2 gap-3">
                            <label className="block text-sm">
                                Model
                                <input
                                    value={openaiModel}
                                    onChange={(e) => setOpenaiModel(e.target.value)}
                                    placeholder="gpt-4o-mini"
                                    className="mt-1 w-full rounded-xl border px-3 py-2"
                                />
                            </label>
                            <div className="flex items-end gap-2">
                                <button
                                    disabled={isGenerating || isRunning}
                                    onClick={startTest}
                                    className="w-full px-3 py-2 rounded-2xl shadow bg-slate-900 text-white disabled:opacity-50"
                                >
                                    {isGenerating ? "Generating…" : "Send 1 turn"}
                                </button>
                            </div>
                        </div>
                        <label className="block text-sm">
                            Base prompt for the 1st message and user context
                            <textarea
                                value={mockPrompt}
                                onChange={(e) => setMockPrompt(e.target.value)}
                                rows={3}
                                className="mt-1 w-full rounded-xl border px-3 py-2"
                            />
                        </label>
                        <div className="flex flex-wrap items-center gap-2">
                            {!isRunning ? (
                                <button
                                    onClick={() => {
                                        setCompletedTurns(0);
                                        runTurns(numTurns);
                                    }}
                                    disabled={!openaiKey}
                                    className="px-3 py-2 rounded-2xl shadow bg-emerald-600 text-white disabled:opacity-50"
                                >
                                    Run {numTurns} turns
                                </button>
                            ) : (
                                <button
                                    onClick={handleStop}
                                    disabled={isStopping}
                                    className="px-3 py-2 rounded-2xl shadow bg-red-600 text-white disabled:opacity-50"
                                >
                                    {isStopping ? "Stopping…" : "Stop"}
                                </button>
                            )}

                            {!isRunning && completedTurns < numTurns && (
                                <button
                                    onClick={handleContinue}
                                    className="px-3 py-2 rounded-2xl shadow bg-slate-900 text-white"
                                >
                                    Continue
                                </button>
                            )}

                            <span className="text-xs text-slate-600 ml-2">
                                {completedTurns}/{numTurns} turns generated
                            </span>
                        </div>
                    </div>
                </section>

                {wsError && (
                    <div className="mb-4 p-3 rounded-xl bg-red-50 text-red-800 border border-red-200">{wsError}</div>
                )}

                <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                    <div className="lg:col-span-2 p-4 rounded-2xl bg-white shadow flex flex-col min-h-[420px]">
                        <h2 className="font-medium mb-3">Conversation</h2>
                        <div ref={scrollRef} className="flex-1 overflow-auto space-y-3 pr-2">
                            {conversation.map((m) => (
                                <MessageBubble key={m.role + m.id} item={m} />
                            ))}
                        </div>
                    </div>

                    <div className="p-4 rounded-2xl bg-white shadow">
                        <h2 className="font-medium mb-3">Metrics (per agent response)</h2>
                        <div className="space-y-2">
                            {Object.values(timings).length === 0 && (
                                <p className="text-sm text-slate-500">No records yet.</p>
                            )}
                            {Object.values(timings)
                                .sort((a, b) => a.requestedAt.localeCompare(b.requestedAt))
                                .map((t) => (
                                    <div key={t.id} className="rounded-xl border p-3">
                                        <div className="text-xs text-slate-500">turn id: {t.id}</div>
                                        <FieldRow label="requestedAt" value={t.requestedAt} />

                                        <FieldRow label="firstDeltaAt" value={t.firstDeltaAt ?? "—"} />
                                        {t.firstDeltaAt && t.requestedAt && (
                                            <FieldRow
                                                label="latency (first - requested)"
                                                value={`${new Date(t.firstDeltaAt).getTime() - new Date(t.requestedAt).getTime()} ms`}
                                            />
                                        )}

                                        <FieldRow label="finalAt" value={t.finalAt ?? "—"} />
                                        {t.firstDeltaAt && t.finalAt && (
                                            <FieldRow
                                                label="latency (final - requested)"
                                                value={`${new Date(t.finalAt).getTime() - new Date(t.requestedAt).getTime()} ms`}
                                            />
                                        )}
                                    </div>
                                ))}
                        </div>
                    </div>
                </section>

                <footer className="mt-8 text-xs text-slate-500">
                    <p>
                        Tip: your agent emits <code>delta</code> and <code>final</code> events. This app aggregates the{" "}
                        <code>response</code> text and stores timestamps for <em>requested</em>, first <em>delta</em>,
                        and <em>final</em>. Extra data (<code>action_payloads</code>, <code>next_step</code>,{" "}
                        <code>next_step_reason</code>) is accessible in the agent bubbles. When running multiple turns,
                        the same <code>thread_id</code> is used to preserve context. Share links include config,
                        history, and timings (never your API key).
                    </p>
                </footer>
            </div>
        </div>
    );
}

function FieldRow({ label, value }: { label: string; value: string }) {
    return (
        <div className="grid grid-cols-3 gap-2 text-sm py-1">
            <div className="text-slate-500">{label}</div>
            <div className="col-span-2 font-mono break-all">{value}</div>
        </div>
    );
}

function MessageBubble({ item }: { item: ConversationItem }) {
    const [showExtra, setShowExtra] = useState(false);
    const isAgent = item.role === "agent";
    return (
        <div className={`flex ${isAgent ? "justify-start" : "justify-end"}`}>
            <div
                className={`max-w-[90%] rounded-2xl px-4 py-3 shadow ${
                    isAgent ? "bg-slate-100" : "bg-slate-900 text-white"
                }`}
            >
                <div className="text-xs opacity-70 mb-1 flex items-center gap-2">
                    <span className="font-medium">{isAgent ? "Agent" : "User"}</span>
                    <span className="font-mono">{item.at}</span>
                    {isAgent && (
                        <button onClick={() => setShowExtra((v) => !v)} className="text-[11px] underline">
                            {showExtra ? "hide extras" : "view extras"}
                        </button>
                    )}
                </div>
                <div className="whitespace-pre-wrap leading-relaxed">
                    {item.text || <span className="opacity-60">(empty)</span>}
                </div>
                {isAgent && showExtra && item.extra && (
                    <pre className="mt-2 text-xs overflow-auto max-h-64 bg-white rounded-xl p-2 border">
                        {JSON.stringify(item.extra, null, 2)}
                    </pre>
                )}
            </div>
        </div>
    );
}
