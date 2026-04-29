<?php
/**
 * ============================================================
 * SAMS Chatbot — PHP Integration
 * ============================================================
 * This file shows how your existing PHP system can communicate
 * with the Python chatbot API.
 *
 * Two approaches:
 *   1. PHP cURL (server-side — recommended for production)
 *   2. JavaScript fetch (client-side — simpler for testing)
 *
 * The Python Flask API must be running first:
 *   python 02_api_server.py
 * ============================================================
 */


// ─────────────────────────────────────────────────────────────
// CONFIGURATION
// ─────────────────────────────────────────────────────────────

define('CHATBOT_API_URL', 'http://localhost:5000/chat');
define('CHATBOT_API_TIMEOUT', 10);  // seconds before giving up


// ─────────────────────────────────────────────────────────────
// APPROACH 1 — PHP cURL (Server-Side)
//
// Your PHP controller sends the request to Python,
// then passes the response back to the student's browser.
// This keeps the Python API URL hidden from users.
// ─────────────────────────────────────────────────────────────

/**
 * Send a message to the SAMS chatbot API and return the result.
 *
 * @param  string $userMessage  The student's raw input text.
 * @param  bool   $useHybrid   Whether to use keyword matching first.
 * @return array               Associative array with 'success', 'response', 'intent', etc.
 */
function callChatbotAPI(string $userMessage, bool $useHybrid = true): array
{
    // Sanitise input before sending
    $userMessage = trim(strip_tags($userMessage));

    if (empty($userMessage)) {
        return [
            'success'  => false,
            'response' => 'Please enter a message.',
            'intent'   => 'unknown',
        ];
    }

    if (strlen($userMessage) > 200) {
        return [
            'success'  => false,
            'response' => 'Your message is too long. Please keep it brief.',
            'intent'   => 'unknown',
        ];
    }

    // Build JSON payload
    $payload = json_encode([
        'message'    => $userMessage,
        'use_hybrid' => $useHybrid,
    ]);

    // Initialise cURL
    $ch = curl_init(CHATBOT_API_URL);

    curl_setopt_array($ch, [
        CURLOPT_POST           => true,
        CURLOPT_POSTFIELDS     => $payload,
        CURLOPT_RETURNTRANSFER => true,
        CURLOPT_TIMEOUT        => CHATBOT_API_TIMEOUT,
        CURLOPT_HTTPHEADER     => [
            'Content-Type: application/json',
            'Content-Length: ' . strlen($payload),
        ],
    ]);

    // Execute request
    $raw      = curl_exec($ch);
    $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    $error    = curl_error($ch);
    curl_close($ch);

    // Handle connection errors
    if ($error || $raw === false) {
        error_log("SAMS Chatbot API error: $error");
        return [
            'success'  => false,
            'response' => 'The chatbot service is temporarily unavailable. Please try again later.',
            'intent'   => 'error',
        ];
    }

    // Decode JSON response
    $data = json_decode($raw, true);

    if ($httpCode !== 200 || !$data || !$data['success']) {
        return [
            'success'  => false,
            'response' => 'Sorry, the chatbot encountered an error. Please try again.',
            'intent'   => 'error',
        ];
    }

    return $data;
}


// ─────────────────────────────────────────────────────────────
// APPROACH 2 — REST Endpoint (for AJAX from your PHP pages)
//
// This acts as a PHP "proxy" controller.
// Your JavaScript calls THIS PHP file, which calls Python.
// ─────────────────────────────────────────────────────────────

/**
 * chat_endpoint.php — include this logic in a dedicated PHP file.
 *
 * Your JS would call:  POST /chatbot/chat_endpoint.php
 *                      Body: { "message": "How do I pay rent?" }
 */
function handleChatRequest(): void
{
    // Only accept POST
    if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
        http_response_code(405);
        echo json_encode(['error' => 'Method not allowed']);
        return;
    }

    // Set JSON response header
    header('Content-Type: application/json');
    header('Access-Control-Allow-Origin: *');   // adjust in production

    // Parse incoming JSON
    $body = file_get_contents('php://input');
    $data = json_decode($body, true);

    if (!$data || !isset($data['message'])) {
        http_response_code(400);
        echo json_encode(['error' => "Field 'message' is required"]);
        return;
    }

    // Optionally: check if user is logged in (SAMS session check)
    // session_start();
    // if (!isset($_SESSION['student_id'])) {
    //     http_response_code(401);
    //     echo json_encode(['error' => 'Unauthorised']);
    //     return;
    // }

    // Call chatbot API
    $result = callChatbotAPI($data['message']);

    // Optionally log to your SAMS database
    // logQueryToDatabase($_SESSION['student_id'], $data['message'], $result['intent']);

    echo json_encode($result);
}


// ─────────────────────────────────────────────────────────────
// DATABASE LOGGING (Optional)
//
// Log student queries to your MySQL database for analytics
// and future model improvement.
// ─────────────────────────────────────────────────────────────

/**
 * Log a chatbot query to the SAMS database.
 *
 * First create this table in your MySQL database:
 *
 *   CREATE TABLE chatbot_logs (
 *     id          INT AUTO_INCREMENT PRIMARY KEY,
 *     student_id  INT NULL,
 *     raw_message TEXT NOT NULL,
 *     intent      VARCHAR(50),
 *     confidence  FLOAT,
 *     method      VARCHAR(20),
 *     created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
 *   );
 *
 * @param  PDO    $pdo        Your existing PDO connection.
 * @param  int    $studentId  Logged-in student's ID (or null for anonymous).
 * @param  string $message    Raw user message.
 * @param  array  $apiResult  Full result from callChatbotAPI().
 */
function logQueryToDatabase(PDO $pdo, ?int $studentId, string $message, array $apiResult): void
{
    $stmt = $pdo->prepare("
        INSERT INTO chatbot_logs (student_id, raw_message, intent, confidence, method)
        VALUES (:student_id, :raw_message, :intent, :confidence, :method)
    ");

    $stmt->execute([
        ':student_id'  => $studentId,
        ':raw_message' => $message,
        ':intent'      => $apiResult['intent']     ?? 'unknown',
        ':confidence'  => $apiResult['confidence'] ?? 0.0,
        ':method'      => $apiResult['method']     ?? 'unknown',
    ]);
}


// ─────────────────────────────────────────────────────────────
// DIRECT USAGE EXAMPLE
//
// Paste this in any PHP file in your SAMS project:
// ─────────────────────────────────────────────────────────────

/*

// --- Simple usage in a PHP page ---
$result   = callChatbotAPI("How do I pay my rent?");
$response = $result['response'];
$intent   = $result['intent'];

echo "<p>Bot: " . htmlspecialchars($response) . "</p>";


// --- AJAX endpoint (save as: /chatbot/chat_endpoint.php) ---
// handleChatRequest();   // uncomment this line

*/


?>
<!-- ─────────────────────────────────────────────────────────────
  STUDENT CHAT WIDGET — HTML + JavaScript
  
  Embed this in any SAMS page (e.g. dashboard.php, home.php).
  It calls the PHP proxy (chat_endpoint.php), which calls Python.
  ──────────────────────────────────────────────────────────────── -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAMS — Student Chat Assistant</title>
    <style>
        /* ── Chat widget styles ── */
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f6fa; }

        #chat-widget {
            position: fixed;
            bottom: 20px; right: 20px;
            width: 360px;
            max-height: 520px;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.15);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            font-size: 14px;
        }

        #chat-header {
            background: #1a4a8a;
            color: #fff;
            padding: 14px 18px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        #chat-header .dot {
            width: 10px; height: 10px;
            background: #4ade80;
            border-radius: 50%;
        }

        #chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 14px;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-height: 360px;
        }

        .msg {
            max-width: 78%;
            padding: 10px 14px;
            border-radius: 12px;
            line-height: 1.5;
            word-wrap: break-word;
        }
        .msg.bot  { background: #eef2ff; color: #1e293b; align-self: flex-start; border-bottom-left-radius: 4px; }
        .msg.user { background: #1a4a8a; color: #fff;    align-self: flex-end;   border-bottom-right-radius: 4px; }

        #chat-input-area {
            display: flex;
            padding: 12px;
            gap: 8px;
            border-top: 1px solid #e2e8f0;
        }

        #chat-input {
            flex: 1;
            padding: 10px 14px;
            border: 1px solid #cbd5e1;
            border-radius: 24px;
            outline: none;
            font-size: 14px;
            transition: border-color 0.2s;
        }
        #chat-input:focus { border-color: #1a4a8a; }

        #chat-send {
            background: #1a4a8a;
            color: #fff;
            border: none;
            border-radius: 50%;
            width: 40px; height: 40px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.2s, transform 0.1s;
            display: flex; align-items: center; justify-content: center;
        }
        #chat-send:hover   { background: #153a6e; }
        #chat-send:active  { transform: scale(0.95); }
        #chat-send:disabled { background: #94a3b8; cursor: not-allowed; }

        .typing { display: flex; align-items: center; gap: 5px; padding: 10px 14px; }
        .typing span { width: 7px; height: 7px; background: #94a3b8; border-radius: 50%; animation: bounce 1.2s ease-in-out infinite; }
        .typing span:nth-child(2) { animation-delay: 0.2s; }
        .typing span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
    </style>
</head>
<body>

<div id="chat-widget">
    <div id="chat-header">
        <div class="dot"></div>
        SAMS Virtual Assistant
    </div>

    <div id="chat-messages">
        <!-- Initial greeting -->
        <div class="msg bot">
            👋 Hello! I'm your SAMS assistant. I can help with payments, accommodation rules, visitor policies, and application status. What would you like to know?
        </div>
    </div>

    <div id="chat-input-area">
        <input
            type="text"
            id="chat-input"
            placeholder="Type your question..."
            maxlength="200"
            autocomplete="off"
        />
        <button id="chat-send" title="Send">&#9658;</button>
    </div>
</div>

<script>
/**
 * SAMS Chat Widget JavaScript
 *
 * Sends user messages to the PHP proxy endpoint,
 * which forwards them to the Python chatbot API.
 *
 * CHANGE this URL to match your SAMS server path:
 */
const CHAT_ENDPOINT = '/chatbot/chat_endpoint.php';

const messagesEl = document.getElementById('chat-messages');
const inputEl    = document.getElementById('chat-input');
const sendBtn    = document.getElementById('chat-send');

/** Add a message bubble to the chat window */
function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
}

/** Show animated typing indicator */
function showTyping() {
    const el = document.createElement('div');
    el.className = 'msg bot typing';
    el.id = 'typing-indicator';
    el.innerHTML = '<span></span><span></span><span></span>';
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

/** Remove typing indicator */
function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

/** Send message to PHP → Python → response */
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    // Show user's message
    addMessage(text, 'user');
    inputEl.value = '';
    sendBtn.disabled = true;
    showTyping();

    try {
        const response = await fetch(CHAT_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        hideTyping();

        if (data.success && data.response) {
            addMessage(data.response, 'bot');
        } else {
            addMessage('Sorry, I ran into an issue. Please try again.', 'bot');
        }

    } catch (error) {
        hideTyping();
        console.error('Chat error:', error);
        addMessage('Connection issue. Please check your connection and try again.', 'bot');
    } finally {
        sendBtn.disabled = false;
        inputEl.focus();
    }
}

// Event listeners
sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) sendMessage();
});
</script>

</body>
</html>
