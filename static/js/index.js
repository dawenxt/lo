// LOGIN.PHP
function showLoginForm() {
    document.getElementById("login-form").style.display = "block";
    document.getElementById("register-form").style.display = "none";
    localStorage.setItem("formView", "login");
    document.querySelector('.btn1').classList.add('active');
    document.querySelector('.btn2').classList.remove('active');
    document.querySelector('.btn1').classList.remove('inactive');
    document.querySelector('.btn2').classList.add('inactive');
}

// Function to show register form
function showRegisterForm() {
    document.getElementById("login-form").style.display = "none";
    document.getElementById("register-form").style.display = "block";
    localStorage.setItem("formView", "register");
    document.querySelector('.btn2').classList.add('active');
    document.querySelector('.btn1').classList.remove('active');
    document.querySelector('.btn2').classList.remove('inactive');
    document.querySelector('.btn1').classList.add('inactive');
}

// Function to initialize form view
function initFormView() {
    var formView = localStorage.getItem("formView");
    if (!formView) {
        localStorage.setItem("formView", "login-form"); 
        showLoginForm(); 
    } else if (formView === "register") {
        showRegisterForm();
    } else {
        showLoginForm();
    }
}


// Function to initialize sliding text position
function initSlidingTextPosition() {
    var slidingText = document.querySelector('.sliding-text');
    var slidingTextContainer = document.querySelector('.sliding-text-container');
    var slidingTextWidth = slidingText.offsetWidth;
    slidingText.style.transform = 'translateX(' + slidingTextWidth + 'px)';
}

// Call initSlidingTextPosition when window loads
window.onload = function() {
    initFormView(); // Initialize form view
    initSlidingTextPosition(); // Initialize sliding text position
};


// Call initFormView when window loads
window.onload = initFormView;



// INDEX.PHP
function openPopup() {
    document.getElementById("popup").style.display = "block";
}

// Function to verify the current password
function verifyPassword() {
    var currentPassword = document.getElementById("current_password").value;

    
    window.location.href = "reset.php";
}

// Function to cancel the password reset
function cancelReset() {
    document.getElementById("popup").style.display = "none"; // Hide the pop-up
    document.getElementById("current_password").value = ""; // Clear the password field
}

