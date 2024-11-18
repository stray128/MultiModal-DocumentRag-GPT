let questions = [];
let results = {};
let currentPdfName = '';

// Initialize PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.10.377/pdf.worker.min.js';

function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.style.display = 'block';
    setTimeout(() => {
        notification.style.display = 'none';
    }, 3000);
}

function displayFileName() {
    const fileInput = document.getElementById('fileInput');
    const fileNameSpan = document.getElementById('fileName');
    if (fileInput.files.length > 0) {
        fileNameSpan.textContent = fileInput.files[0].name;
    }
}

function showLoadingOverlay(message) {
    const overlay = document.getElementById('loadingOverlay');
    const messageElement = document.getElementById('loadingMessage');
    messageElement.textContent = message;
    overlay.style.display = 'flex';
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'none';
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showNotification('Please select a file first', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showLoadingOverlay('Uploading and processing PDF...');
        
        const response = await fetch('/upload/', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        hideLoadingOverlay();
        showNotification('File uploaded and processed successfully');
        currentPdfName = file.name;
        
        // Clear the file input
        fileInput.value = '';
        document.getElementById('fileName').textContent = '';
        
    } catch (error) {
        hideLoadingOverlay();
        console.error('Error:', error);
        showNotification('Error uploading file: ' + error.message, 'error');
    }
}

async function renderPDF(highlights) {
    try {
        const viewer = document.getElementById('pdfViewer');
        viewer.innerHTML = '';

        if (!highlights || highlights.length === 0) {
            console.error('No highlights provided');
            return;
        }

        const filename = highlights[0].filename;
        const pdfUrl = window.location.origin + `/uploads/${filename}`;

        console.log("Loading PDF from URL:", pdfUrl);

        const loadingTask = pdfjsLib.getDocument(pdfUrl);
        const pdf = await loadingTask.promise;

        const renderPage = async (pageNum) => {
            const page = await pdf.getPage(pageNum);
            const viewport = page.getViewport({ scale: 1 });
            const scale = viewer.clientWidth / viewport.width; // Scale based on viewer width
            const scaledViewport = page.getViewport({ scale });

            const pageContainer = document.createElement('div');
            pageContainer.className = 'page-container';
            
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = scaledViewport.height;
            canvas.width = scaledViewport.width;

            pageContainer.appendChild(canvas);
            viewer.appendChild(pageContainer);

            await page.render({
                canvasContext: context,
                viewport: scaledViewport
            }).promise;
        };

        // Render all pages first
        for (let i = 1; i <= pdf.numPages; i++) {
            await renderPage(i);
        }

        // Add highlights after rendering with a delay
        setTimeout(() => {
            const firstHighlightPage = highlights[0].page_number;
            const pageContainers = viewer.getElementsByClassName('page-container');
            
            if (pageContainers[firstHighlightPage - 1]) {
                pageContainers[firstHighlightPage - 1].scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'start'
                });

                // Add highlights after scrolling
                setTimeout(() => {
                    highlights.forEach(highlight => {
                        const pageContainer = pageContainers[highlight.page_number - 1];
                        if (pageContainer) {
                            const [x1, y1, x2, y2] = highlight.bbox;
                            const layout_width = highlight.layout_width;
                            const layout_height = highlight.layout_height;
                            
                            const highlightDiv = document.createElement('div');
                            highlightDiv.className = 'highlight';
                            highlightDiv.style.position = 'absolute';
                            highlightDiv.style.left = `${(x1 / layout_width) * pageContainer.clientWidth}px`;
                            highlightDiv.style.top = `${(y1 / layout_height) * pageContainer.clientHeight}px`;
                            highlightDiv.style.width = `${((x2 - x1) / layout_width) * pageContainer.clientWidth}px`;
                            highlightDiv.style.height = `${((y2 - y1) / layout_height) * pageContainer.clientHeight}px`;
                            pageContainer.appendChild(highlightDiv);
                        }
                    });
                }, 1500);
            }
        }, 500);

    } catch (error) {
        console.error('Error rendering PDF:', error);
        showNotification('Error rendering PDF: ' + error.message, 'error');
    }
}

function addQuestion() {
    const input = document.getElementById('questionInput');
    if (input.value.trim()) {
        questions.push(input.value.trim());
        updateQuestionList();
        input.value = '';
        showNotification('Question added successfully');
    }
}

async function handleFileUpload() {
    const file = document.getElementById('questionsFile').files[0];
    if (!file) {
        showNotification('Please select a file first', 'error');
        return;
    }

    const fileExtension = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'xlsx'].includes(fileExtension)) {
        showNotification('Please upload a CSV or XLSX file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload_questions/', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        if (result.status === 'success') {
            questions = result.questions;
            updateQuestionList();
            showNotification(`Loaded ${questions.length} questions successfully`);
        } else {
            showNotification(result.message || 'Error loading questions', 'error');
        }
    } catch (error) {
        showNotification('Error uploading file: ' + error.message, 'error');
    }
}

async function processAllQuestions() {
    if (questions.length === 0) {
        showNotification('No questions to process', 'error');
        return;
    }

    try {
        showLoadingOverlay('Processing questions...');
        
        questions.forEach(question => {
            results[question] = { loading: true };
        });
        updateQuestionList();

        const response = await fetch('/multi_query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                questions: questions,
                filename: currentPdfName
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Multi-query response:", data);
        
        data.results.forEach(resultObj => {
            const question = resultObj.question;
            if (resultObj.error) {
                results[question] = { error: resultObj.error };
            } else {
                results[question] = resultObj.result;
            }
        });

        hideLoadingOverlay();
        updateQuestionList();
        showNotification('All questions processed successfully');

    } catch (error) {
        hideLoadingOverlay();
        console.error('Error processing questions:', error);
        showNotification('Error processing questions: ' + error.message, 'error');
        
        questions.forEach(question => {
            results[question] = { error: error.message };
        });
        updateQuestionList();
    }
}

function updateQuestionList() {
    const container = document.getElementById('questionList');
    container.innerHTML = '';
    
    questions.forEach((question, index) => {
        const result = results[question];
        const div = document.createElement('div');
        div.className = 'question-item';
        
        let resultHtml = '';
        if (result) {
            if (result.loading) {
                resultHtml = '<div class="answer-section">Loading...</div>';
            } else if (result.error) {
                resultHtml = `<div class="answer-section error">${result.error}</div>`;
            } else {
                resultHtml = `
                    <div class="answer-section" id="answer-${index}" style="display: none;">
                        <div class="result-header">
                            <div class="final-answer">
                                <h4>Answer:</h4>
                                <p>${marked.parse(result.result)}</p>
                            </div>
                            <button class="styled-button" onclick="showContext(${index})" style="margin-top: 12px;">
                                Show Context
                            </button>
                        </div>
                    </div>
                `;
            }
        }
        
        div.innerHTML = `
            <div class="question-header" onclick="toggleAnswer(${index})">
                <span class="question-text">${question}</span>
                <span class="toggle-icon">▼</span>
            </div>
            ${resultHtml}
        `;
        
        container.appendChild(div);
    });
}

function toggleAnswer(index) {
    const answerSection = document.getElementById(`answer-${index}`);
    const currentDisplay = answerSection.style.display;
    
    // Close all answer sections
    document.querySelectorAll('.answer-section').forEach(section => {
        section.style.display = 'none';
    });
    
    // Open the clicked section if it was closed
    if (currentDisplay === 'none' || currentDisplay === '') {
        answerSection.style.display = 'block';
    }
}

async function showContext(index) {
    const question = questions[index];
    const result = results[question];

    console.log("Result:", result);
    console.log("Generated questions:", result.generated_questions);

    if (!result || !result.metadata) {
        console.log("Metadata missing. Result structure:", {
            exists: !!result,
            hasMetadata: result ? !!result.metadata : false,
            keys: result ? Object.keys(result) : []
        });
        showNotification('No context available for this question', 'error');
        return;
    }
    
    try {
        // Since we're dealing with a single question's result, we don't need to index the metadata
        const contextData = {
            metadata: {
                sources: result.metadata.sources,
                ranked_metadata: result.metadata.ranked_metadata
            }
        };

        console.log("Sending context data:", contextData);

        const response = await fetch('/get_context/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(contextData)
        });
        
        const contextResult = await response.json();
        console.log("Context API response:", contextResult);

        if (contextResult.highlights && contextResult.highlights.length > 0) {
            await renderPDF(contextResult.highlights);
            showNotification('Context loaded successfully');
        } else {
            showNotification('No context found', 'error');
        }
    } catch (error) {
        console.error('Error loading context:', error);
        showNotification('Error loading context: ' + error.message, 'error');
    }
}

function clearFileInput(inputId) {
    const fileInput = document.getElementById(inputId);
    fileInput.value = '';
    if (inputId === 'fileInput') {
        document.getElementById('fileName').textContent = '';
        document.getElementById('uploadResult').textContent = '';
    }
}

function clearSession() {
    // Clear all data
    questions = [];
    results = {};
    currentPdfName = '';
    
    // Clear UI elements
    document.getElementById('fileInput').value = '';
    document.getElementById('fileName').textContent = '';
    document.getElementById('uploadResult').textContent = '';
    document.getElementById('questionsFile').value = '';
    document.getElementById('questionInput').value = '';
    document.getElementById('questionList').innerHTML = '';
    document.getElementById('pdfViewer').innerHTML = '';
    
    // Show notification
    showNotification('Session cleared successfully');
}

// Initialize PDF.js when the page loads
window.onload = function() {
    // Add your PDF.js initialization code here
};

async function submitQuestion() {
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question) {
        showNotification('Please enter a question', 'error');
        return;
    }

    try {
        showLoadingState();
        
        const response = await fetch('/query/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                filename: currentPdfName
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        
        // Debug logs
        console.log("Full API Response:", result);
        console.log("Response structure:", {
            hasResult: !!result.result,
            hasMetadata: !!result.metadata,
            metadataKeys: result.metadata ? Object.keys(result.metadata) : [],
            fullMetadata: result.metadata
        });
        
        // Store question and result
        questions.push(question);
        results[question] = result;  // Store the complete result object
        
        updateQuestionList();
        clearLoadingState();
        
    } catch (error) {
        console.error('Error:', error);
        showNotification('Error submitting question: ' + error.message, 'error');
        clearLoadingState();
    }
}

function downloadResults() {
    const resultsToDownload = {};
    questions.forEach(question => {
        if (results[question] && !results[question].loading && !results[question].error) {
            resultsToDownload[question] = results[question].result;
        }
    });

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(resultsToDownload, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "query_results.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
    showNotification('Results downloaded successfully');
}