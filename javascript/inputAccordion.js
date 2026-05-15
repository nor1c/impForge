function inputAccordionChecked(id, checked) {
    var accordion = gradioApp().getElementById(id);
    if (!accordion || !accordion.visibleCheckbox) return;
    accordion.visibleCheckbox.checked = checked;
    accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
    var labelWrap = accordion.querySelector('.label-wrap');
    var gradioCheckbox = gradioApp().querySelector('#' + accordion.id + "-checkbox input");
    var extra = gradioApp().querySelector('#' + accordion.id + "-extra");
    var span = labelWrap.querySelector('span');
    var linked = true;
    var skipInitial = accordion.classList.contains('input-accordion-skip-initial');
    var initialSyncDone = !skipInitial;

    var isOpen = function() {
        return labelWrap.classList.contains('open');
    };

    var observerAccordionOpen = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutationRecord) {
            accordion.classList.toggle('input-accordion-open', isOpen());

            if (linked) {
                accordion.visibleCheckbox.checked = isOpen();
                accordion.onVisibleCheckboxChange();
            }
        });
    });
    observerAccordionOpen.observe(labelWrap, {attributes: true, attributeFilter: ['class']});

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    accordion.onChecked = function(checked) {
        if (isOpen() != checked) {
            labelWrap.click();
        }
    };

    var visibleCheckbox = document.createElement('INPUT');
    visibleCheckbox.type = 'checkbox';
    // Seed from the underlying gradio checkbox state so we honor decoupled
    // (value=true, open=false) initial configurations without forcing the
    // accordion to expand.
    visibleCheckbox.checked = skipInitial ? gradioCheckbox.checked : isOpen();
    visibleCheckbox.id = accordion.id + "-visible-checkbox";
    visibleCheckbox.className = gradioCheckbox.className + " input-accordion-checkbox";
    span.insertBefore(visibleCheckbox, span.firstChild);

    accordion.visibleCheckbox = visibleCheckbox;
    accordion.onVisibleCheckboxChange = function() {
        if (!initialSyncDone) {
            initialSyncDone = true;
            // Don't auto-toggle the accordion on the first sync when the
            // caller intentionally decoupled value from open state. Just
            // propagate the value to the underlying gradio checkbox.
            gradioCheckbox.checked = visibleCheckbox.checked;
            updateInput(gradioCheckbox);
            return;
        }

        if (linked && isOpen() != visibleCheckbox.checked) {
            labelWrap.click();
        }

        gradioCheckbox.checked = visibleCheckbox.checked;
        updateInput(gradioCheckbox);
    };

    visibleCheckbox.addEventListener('click', function(event) {
        linked = false;
        event.stopPropagation();
    });
    visibleCheckbox.addEventListener('input', accordion.onVisibleCheckboxChange);
}

onUiLoaded(function() {
    for (var accordion of gradioApp().querySelectorAll('.input-accordion')) {
        setupAccordion(accordion);
    }
});
