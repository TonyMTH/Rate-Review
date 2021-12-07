import streamlite as st

from predict import RateReview

# Text/Title
st.title("Rate Review")

review_title = st.text_input("Review Title")
review_body = st.text_area("Review Body")

model_pth = 'data/best_model.pt'

if st.button('Rate'):
    if review_title is not None and review_body is not None:
        review = RateReview(model_pth).predict(review_title, review_body)
        st.write(str(review)+" stars")
    else:
        st.write("None of the field should be empty")

