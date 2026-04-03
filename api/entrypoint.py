result = service.identify_and_prepare(
    image_path="data/input/IMG_9309.JPG",
    project_id=1,
    top_k=5,
    debug=True
)

print(result)
decision = input("Choose: NEW | MATCH | CANCEL >>> ")
if decision == "MATCH":
    id = input("ID: >>> ")
    print(service.confirm_decision(
        upload_id=result["upload_id"],
        decision=decision,
        existing_card_id=id,
        card_data={
            "status": "idfk",
            "water_body_number": "921",
            "length_body": "3",
            "length_tail": "6"
        }
    ))