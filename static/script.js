// Fetch the detected word and update the display
function fetchDetectedWord() {
    fetch('/get_prediction')
        .then(response => response.json())
        .then(data => {
            const detectedWord = document.getElementById('detected-word');
            detectedWord.textContent = data.prediction || '_';
        })
        .catch(error => console.error('Error fetching detected word:', error));
}

// Fetch the current letter and update the display
function fetchCurrentLetter() {
    fetch('/get_prediction')
        .then(response => response.json())
        .then(data => {
            const currentLetter = document.getElementById('gesture');
            currentLetter.textContent = data.current_letter || '-';
        })
        .catch(error => console.error('Error fetching current letter:', error));
}

// Function to add space by calling the backend
function addSpace() {
    fetch('/update_word', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: 'space' })
    })
    .then(response => response.json())
    .then(data => {
        fetchDetectedWord();  // Update the word display after the change
    })
    .catch(error => console.error('Error updating word:', error));
}

// Function to delete the last character by calling the backend
function deleteLast() {
    fetch('/update_word', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ action: 'delete' })
    })
    .then(response => response.json())
    .then(data => {
        fetchDetectedWord();  // Update the word display after the change
    })
    .catch(error => console.error('Error updating word:', error));
}

// Fetch the detected word every 2 seconds to keep display updated
setInterval(fetchDetectedWord, 2000);
// Fetch the current letter every 2 seconds to keep display updated
setInterval(fetchCurrentLetter, 2000);
